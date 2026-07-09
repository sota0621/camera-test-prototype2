const video = document.getElementById('camera-view');
const canvas = document.getElementById('canvas-output');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('start-btn');
const stopRecordBtn = document.getElementById('stop-record-btn');
const resetConfigBtn = document.getElementById('reset-config-btn'); // 追加
const statusText = document.getElementById('status');

let src, dst, hsv, mask, contours, hierarchy;
let isProcessing = false;

let lockCounter = 0;
const REQUIRED_FRAMES = 150; // 5秒間安定

// ブレ対策：直近30フレームの履歴を記憶するバッファ（打率制）
const BUFFER_SIZE = 30;
let detectionHistory = new Array(BUFFER_SIZE).fill(false);
let historyIndex = 0;

let mediaRecorder;
let isRecording = false;
let cameraStream = null;
let animationFrameId = null;

// --- 【新規追加】画角マスター座標管理用の変数とキー ---
const STORAGE_KEY = "experiment_master_pts";
let masterPoints = JSON.parse(localStorage.getItem(STORAGE_KEY)) || null;

// --- IndexedDB の初期化設定 ---
const DB_NAME = "ExperimentVideoDB";
const STORE_NAME = "video_chunks";
let db = null;

function initIndexedDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(DB_NAME, 1);
        request.onupgradeneeded = (e) => {
            const database = e.target.result;
            if (!database.objectStoreNames.contains(STORE_NAME)) {
                database.createObjectStore(STORE_NAME, { autoIncrement: true });
            }
        };
        request.onsuccess = (e) => {
            db = e.target.result;
            resolve();
        };
        request.onerror = (e) => reject(e);
    });
}

// --- 【新規追加】ステータスバッジの文言と色を更新する関数 ---
function updateStatusMessage() {
    statusText.classList.remove('loading', 'ready', 'recording');
    if (isRecording) {
        statusText.classList.add('recording');
        statusText.innerHTML = `🔴 長時間実験映像を録画中...`;
    } else if (masterPoints) {
        statusText.classList.add('ready');
        statusText.innerHTML = `📅 【2回目以降】初回の黄色枠に合わせてください`;
    } else {
        statusText.classList.add('ready');
        statusText.innerHTML = `🆕 【初回撮影】理想の画角に設置してください`;
    }
}

// --- 【新規追加】リセットボタンのクリックイベント ---
resetConfigBtn.addEventListener('click', () => {
    if (confirm("保存されている初回の画角データを削除し、新しく作り直しますか？")) {
        localStorage.removeItem(STORAGE_KEY);
        masterPoints = null;
        lockCounter = 0;
        canvas.classList.remove('locked');
        updateStatusMessage();
    }
});

// --- 【新規追加】現在の4点座標が初回の座標と一致しているか判定する関数 ---
function isSamePoints(ptsA, ptsB, threshold = 20) {
    if (!ptsA || !ptsB || ptsA.length !== 4 || ptsB.length !== 4) return false;
    for (let i = 0; i < 4; i++) {
        const dx = ptsA[i].x - ptsB[i].x;
        const dy = ptsA[i].y - ptsB[i].y;
        if (Math.sqrt(dx * dx + dy * dy) > threshold) {
            return false; // 1点でも許容ピクセル（20px）を超えてズレていたら不可
        }
    }
    return true;
}

// カメラ起動と初期化
async function startSystem() {
    try {
        await initIndexedDB();
        
        const constraints = {
            video: {
                facingMode: { ideal: "environment" },
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        };

        cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = cameraStream;
        await video.play();

        const track = cameraStream.getVideoTracks()[0];
        const capabilities = track.getCapabilities();
        const settings = track.getSettings();
        
        let targetId = null;
        if (capabilities.facingMode) {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = devices.filter(d => d.kind === 'videoinput');
            const ultraWide = videoDevices.find(d => 
                d.label.toLowerCase().includes('ultra') || 
                d.label.toLowerCase().includes('0.5') ||
                d.label.toLowerCase().includes('wide')
            );
            if (ultraWide) targetId = ultraWide.deviceId;
        }

        if (targetId && settings.deviceId !== targetId) {
            cameraStream.getTracks().forEach(t => t.stop());
            constraints.video.deviceId = { exact: targetId };
            cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
            video.srcObject = cameraStream;
            await video.play();
        }

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        src = new cv.Mat(video.videoHeight, video.videoWidth, cv.CV_8UC4);
        dst = new cv.Mat(video.videoHeight, video.videoWidth, cv.CV_8UC4);
        hsv = new cv.Mat(video.videoHeight, video.videoWidth, cv.CV_8UC3);
        mask = new cv.Mat(video.videoHeight, video.videoWidth, cv.CV_8U);
        contours = new cv.MatVector();
        hierarchy = new cv.Mat();

        isProcessing = true;
        updateStatusMessage();
        processVideo();

    } catch (err) {
        console.error("システム起動エラー:", err);
        statusText.innerText = "カメラの起動に失敗しました。";
    }
}

function processVideo() {
    if (!isProcessing) return;

    try {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        let imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        src.data.set(imageData.data);

        cv.cvtColor(src, dst, cv.COLOR_RGBA2RGB);
        cv.cvtColor(dst, hsv, cv.COLOR_RGB2HSV);

        let low = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [35, 60, 60, 0]);
        let high = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [75, 255, 255, 0]);
        cv.inRange(hsv, low, high, mask);
        low.delete(); high.delete();

        cv.findContours(mask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

        let allCandidates = [];
        for (let i = 0; i < contours.size(); ++i) {
            let cnt = contours.get(i);
            let area = cv.contourArea(cnt);

            // ★バグ修正: 条件式を area > 500 に変更（ノイズカットを正常化）
            if (area > 500) { 
                let perimeter = cv.arcLength(cnt, true);
                let approx = new cv.Mat();
                cv.approxPolyDP(cnt, approx, 0.04 * perimeter, true);

                let moments = cv.moments(cnt, false);
                if (moments.m00 !== 0) {
                    let cx = moments.m01 / moments.m00;
                    let cy = moments.m10 / moments.m00;
                    allCandidates.push({ area: area, x: cy, y: cx });
                }
                approx.delete();
            }
            cnt.delete();
        }

        allCandidates.sort((a, b) => b.area - a.area);
        let validCenters = allCandidates.slice(0, 4);

        // --- 【新規追加】2回目以降モード：初回の「ゴーストガイド枠（黄色）」を描画 ---
        if (masterPoints) {
            ctx.strokeStyle = 'rgba(255, 204, 0, 0.5)'; // 透明度50%の黄色
            ctx.lineWidth = 4;
            ctx.setLineDash([10, 10]); // 点線
            ctx.beginPath();
            ctx.moveTo(masterPoints[0].x, masterPoints[0].y);
            ctx.lineTo(masterPoints[1].x, masterPoints[1].y);
            ctx.lineTo(masterPoints[2].x, masterPoints[2].y);
            ctx.lineTo(masterPoints[3].x, masterPoints[3].y);
            ctx.closePath();
            ctx.stroke();
            ctx.setLineDash([]); // 点線設定を戻す
        }

        if (validCenters.length === 4) {
            validCenters.sort((a, b) => a.y - b.y);
            let topTwo = validCenters.slice(0, 2).sort((a, b) => a.x - b.x);
            let bottomTwo = validCenters.slice(2, 4).sort((a, b) => b.x - a.x);
            let pts = [topTwo[0], topTwo[1], bottomTwo[0], bottomTwo[1]];

            // 現在の緑枠の描画
            ctx.strokeStyle = '#34c759'; ctx.lineWidth = 5;
            ctx.beginPath(); ctx.moveTo(pts[0].x, pts[0].y);
            ctx.lineTo(pts[1].x, pts[1].y); ctx.lineTo(pts[2].x, pts[2].y);
            ctx.lineTo(pts[3].x, pts[3].y); ctx.closePath(); ctx.stroke();

            // --- 【新規追加】画角の一致・安全判定ロジック ---
            let isPositionOK = false;
            if (!masterPoints) {
                // 初回モード：4つの点が見つかっていれば無条件で配置OK
                isPositionOK = true;
            } else {
                // 2回目以降：現在の4点が初回の位置とだいたい同じかチェック
                isPositionOK = isSamePoints(pts, masterPoints, 20); // 許容誤差20px
            }

            if (isPositionOK) {
                lockCounter++;
                if (lockCounter >= REQUIRED_FRAMES) {
                    // ★初回録画スタートの瞬間であれば、現在の4点をマスターとして保存
                    if (!masterPoints) {
                        localStorage.setItem(STORAGE_KEY, JSON.stringify(pts));
                        masterPoints = pts;
                    }

                    src.delete(); dst.delete(); hsv.delete(); mask.delete(); contours.delete(); hierarchy.delete();
                    
                    isProcessing = false;
                    if (animationFrameId) cancelAnimationFrame(animationFrameId);

                    canvas.classList.add('locked');
                    isRecording = true;
                    updateStatusMessage();

                    startRecordingSystem();
                    return; 
                } else {
                    let timeLeft = Math.ceil((REQUIRED_FRAMES - lockCounter) / 30);
                    if (masterPoints) {
                        statusText.innerHTML = `🟡 位置一致中... 自動録画まであと <span style="color: #ffcc00; font-size: 20px;">${timeLeft}</span> 秒`;
                    } else {
                        statusText.innerHTML = `🟡 4点捕捉中... 画角安定まであと <span style="color: #ffcc00; font-size: 20px;">${timeLeft}</span> 秒`;
                    }
                }
            } else {
                lockCounter = 0;
                statusText.innerHTML = `⚠️ 4点検知中：初回の画角（黄色点線）からズレています。位置を合わせてください。`;
            }
        } else {
            lockCounter = 0;
            canvas.classList.remove('locked');
            if (masterPoints) {
                statusText.innerHTML = `🔍 マーカーを探索中... (${validCenters.length} / 4) 黄色い点線に重ねてください`;
            } else {
                statusText.innerHTML = `🔍 マーカーを探しています... (${validCenters.length} / 4)`;
            }
            ctx.fillStyle = 'red';
            for(let p of validCenters) {
                ctx.beginPath(); ctx.arc(p.x, p.y, 8, 0, 2 * Math.PI); ctx.fill();
            }
        }
    } catch (err) {
        console.error(err);
    }

    animationFrameId = requestAnimationFrame(processVideo);
}

// 録画停止ボタンのイベント（既存のロジックを継続）
stopRecordBtn.addEventListener('click', () => {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        isProcessing = false;
        if (animationFrameId) cancelAnimationFrame(animationFrameId);
        
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
        }
        
        statusText.classList.remove('recording');
        statusText.classList.add('loading');
        statusText.innerText = "録画を停止しました。データを保存しています...";
    }
});

// 録画システム開始の処理（既存のMediaRecorder/Chunk保存の仕組みをそのまま呼び出し）
function startRecordingSystem() {
    // あなたが元々構築した、IndexedDBへ1秒ごとにチャンクを書き込む
    // MediaRecorder関連のロジックがここにそのまま入ります。
}

startBtn.addEventListener('click', () => {
    startBtn.disabled = true;
    startSystem();
});

// OpenCVロード完了時
window.onOpenCvReady = () => {
    startBtn.disabled = false;
    resetConfigBtn.style.display = "inline-block"; // リセットボタンを表示
    updateStatusMessage();
};
if (typeof cv !== 'undefined') window.onOpenCvReady();
