import { pipeline, env } from "@huggingface/transformers";

// 設定環境，確保在 Vercel 上能正確抓取模型
env.allowLocalModels = false;

let segmenter;

// 建立帶有回退機制的模型載入函數
const getSegmenter = async () => {
    if (segmenter) return segmenter;

    try {
        console.log("🚀 嘗試啟動 WebGPU 加速...");
        segmenter = await pipeline('image-segmentation', 'briaai/RMBG-1.4', {
            device: 'webgpu',
        });
        console.log("✅ WebGPU 啟動成功！");
    } catch (e) {
        console.warn("⚠️ WebGPU 失敗，正在自動回退到 CPU (WASM) 模式...", e);
        segmenter = await pipeline('image-segmentation', 'briaai/RMBG-1.4', {
            device: 'wasm',
        });
        console.log("ℹ️ 已成功切換至 CPU 模式。");
    }
    return segmenter;
};

// 監聽來自 App.jsx 的訊息
self.onmessage = async (event) => {
    const { img } = event.data;
    if (!img) return;

    try {
        const model = await getSegmenter();
        
        // 執行去背運算
        const output = await model(img);

        // 將結果傳回給主介面
        self.postMessage({ status: 'complete', output });
    } catch (error) {
        self.postMessage({ status: 'error', error: error.message });
    }
};