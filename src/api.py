from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from typing import Dict
import time
import os
import queue
import threading
from f5_tts.api import F5TTS
from importlib.resources import files


# ========== 1. 全局对象：队列 & 任务状态表 ==========
task_queue = queue.Queue()
# 记录任务状态: {task_id: {"status": "pending/in_progress/done/error", "filepath": "xxx"}}
task_status: Dict[str, Dict] = {}

# 这里放音频文件的存储目录
AUDIO_DIR = "output"
os.makedirs(AUDIO_DIR, exist_ok=True)

app = FastAPI()

# ========== 2. 后台线程：从队列中取任务，执行 TTS 函数 ==========
def background_worker():
    while True:
        task_id, text, file_path = task_queue.get()  # 阻塞直到拿到一个任务
        # 更新状态 -> in_progress
        task_status[task_id]["status"] = "in_progress"
        try:
            # 调用TTS函数（示例中只是模拟耗时）
            tts_function(text, file_path)
            # 成功 -> done
            task_status[task_id]["status"] = "done"
        except Exception as e:
            # 失败 -> error
            task_status[task_id]["status"] = "error"
        finally:
            task_queue.task_done()

# 先起一个后台线程来跑
worker_thread = threading.Thread(target=background_worker, daemon=True)
worker_thread.start()


# ========== 3. 模拟TTS函数（在这里替换成你真正的TTS逻辑） ==========
f5tts = F5TTS(
    ckpt_file="../ckpts/mumu_last_reduced2.pt",
    device="mps",
)

def tts_function(text: str, file_path: str):
    try:
        f5tts.infer(
            ref_file=str(files("f5_tts").joinpath("infer/examples/mumu/demo.wav")),
            ref_text="当在观看的疼痛达到最难忍受的高峰时，若能熬得过，例如再忍五分钟，你将发现这椎心刺骨、威胁生命的疼痛开始消退。",
            gen_text=text,
            file_wave=file_path,
            nfe_step=6,
            seed=-1,
        )
    except Exception as e:
        raise "生成失败" + e


# ========== 4. 提供的API接口 ==========
@app.post("/submit_tts_job")
def submit_tts_job(request: dict = Body(...)):
    """
    提交文本合成任务 -> 返回task_id
    """
    text = request.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="文本内容不能为空")
        
    # 用时间戳来简单生成task_id
    task_id = str(int(time.time() * 1000))
    wav_file_path = os.path.join(AUDIO_DIR, f"{task_id}.wav")

    # 初始状态
    task_status[task_id] = {
        "status": "pending", 
        "filepath": wav_file_path
    }
    # 扔到队列中
    task_queue.put((task_id, text, wav_file_path))

    return {"task_id": task_id}


@app.get("/query_tts_status")
def query_tts_status(task_id: str):
    """
    查询任务状态 -> pending / in_progress / done / error / not_found
    如果done，则返回音频文件的下载路径
    """
    info = task_status.get(task_id)
    if not info:
        return {"status": "not_found"}
    status = info["status"]
    resp = {"status": status}
    if status == "done":
        # 构建一个下载音频的URL
        filename = os.path.basename(info["filepath"])
        audio_url = f"/audio/{filename}"
        resp["audio_url"] = audio_url
    return resp


@app.get("/audio/{filename}")
def get_audio(filename: str):
    """
    下载/获取音频文件
    """
    file_path = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"detail": "File not found."})
    return FileResponse(path=file_path, media_type="audio/wav")
