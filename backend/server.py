"""Al-Ansar Voice Cloning - API server (run on a GPU machine / Colab)."""
import os, uuid, time, shutil, threading, queue, traceback, collections
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pipeline

KEY = os.environ.get("ALANSAR_KEY", "")
ROOT = os.path.join(pipeline.WORK, "jobs")
os.makedirs(ROOT, exist_ok=True)

app = FastAPI(title="Al-Ansar Voice Cloning API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MAX_MB = int(os.environ.get("ALANSAR_MAX_MB", "60"))
MAX_JOBS_HOUR = int(os.environ.get("ALANSAR_MAX_JOBS_HOUR", "40"))
RECENT = collections.deque()
STATE = {"ready": False, "error": ""}
_started = False
JOBS = {}
Q = queue.Queue()


def auth(x_key: str = Header(default="")):
    if KEY and x_key != KEY:
        raise HTTPException(status_code=401, detail="کلید غلط ہے")


def _loader():
    try:
        pipeline.load_models()
        STATE["ready"] = True
    except Exception as e:  # noqa
        traceback.print_exc()
        STATE["error"] = str(e)[:300]


def _worker():
    while True:
        jid = Q.get()
        job = JOBS[jid]
        job.update(status="running", message="شروع ہو رہا ہے")
        try:
            def cb(p, m):
                job.update(progress=round(p, 3), message=m)
            r = pipeline.process(job["input"], job["src"], job["tgt"], job["mode"], job["sample"],
                                 job["quality"], job["timing"], cb, job["dir"])
            job.update(status="done", progress=1.0, message="مکمل", audio=r["audio"], video=r["video"],
                       source_text=r["source_text"], target_text=r["target_text"])
        except pipeline.UserError as e:
            job.update(status="error", message=str(e))
        except Exception as e:  # noqa
            traceback.print_exc()
            job.update(status="error", message="خرابی: " + str(e)[:200])


def _cleanup():
    now = time.time()
    for d in os.listdir(ROOT):
        p = os.path.join(ROOT, d)
        if os.path.isdir(p) and now - os.path.getmtime(p) > 3 * 3600:
            shutil.rmtree(p, ignore_errors=True)
            JOBS.pop(d, None)


def start_background(load_models=True):
    global _started
    if _started:
        return
    _started = True
    if load_models:
        threading.Thread(target=_loader, daemon=True).start()
    threading.Thread(target=_worker, daemon=True).start()


@app.on_event("startup")
def _startup():
    if os.environ.get("ALANSAR_NO_AUTOSTART") != "1":
        start_background(True)


async def _save(up: UploadFile, dest: str):
    size = 0
    with open(dest, "wb") as f:
        while True:
            chunk = await up.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_MB * 1048576:
                f.close()
                os.remove(dest)
                raise HTTPException(status_code=413, detail="فائل بہت بڑی ہے (حد %d MB)" % MAX_MB)
            f.write(chunk)


@app.get("/api/health", dependencies=[Depends(auth)])
def health():
    return {"ok": True, "ready": STATE["ready"], "error": STATE["error"]}


@app.post("/api/jobs", dependencies=[Depends(auth)])
async def create_job(file: UploadFile = File(...), src: str = Form("ur"), tgt: str = Form("ar"),
                     mode: str = Form("emotion"), quality: str = Form("best"), timing: str = Form("1"),
                     sample: UploadFile = File(None)):
    if not STATE["ready"]:
        raise HTTPException(status_code=503, detail="ماڈل ابھی لوڈ ہو رہے ہیں")
    _cleanup()
    now = time.time()
    while RECENT and now - RECENT[0] > 3600:
        RECENT.popleft()
    if len(RECENT) >= MAX_JOBS_HOUR:
        raise HTTPException(status_code=429, detail="ایک گھنٹے کی حد پوری ہو گئی، کچھ دیر بعد کوشش کریں")
    RECENT.append(now)
    jid = uuid.uuid4().hex[:10]
    d = os.path.join(ROOT, jid)
    os.makedirs(d, exist_ok=True)
    ext = os.path.splitext(file.filename or "")[1][:8] or ".bin"
    inp = os.path.join(d, "input" + ext)
    await _save(file, inp)
    smp = None
    if sample is not None and sample.filename:
        smp = os.path.join(d, "sample_in" + (os.path.splitext(sample.filename)[1][:8] or ".bin"))
        await _save(sample, smp)
    JOBS[jid] = dict(status="queued", progress=0.0, message="قطار میں", input=inp, sample=smp, dir=d,
                     src=src, tgt=tgt, mode=mode, quality=quality, timing=(timing == "1"),
                     audio=None, video=None, source_text="", target_text="")
    Q.put(jid)
    return {"id": jid}


@app.get("/api/jobs/{jid}", dependencies=[Depends(auth)])
def job_status(jid: str):
    j = JOBS.get(jid)
    if not j:
        raise HTTPException(status_code=404, detail="کام نہیں ملا")
    return {"status": j["status"], "progress": j["progress"], "message": j["message"],
            "source_text": j["source_text"], "target_text": j["target_text"], "has_video": bool(j["video"])}


@app.get("/api/jobs/{jid}/audio", dependencies=[Depends(auth)])
def job_audio(jid: str):
    j = JOBS.get(jid)
    if not j or not j["audio"]:
        raise HTTPException(status_code=404, detail="تیار نہیں")
    return FileResponse(j["audio"], media_type="audio/wav", filename="AlAnsar.wav")


@app.get("/api/jobs/{jid}/video", dependencies=[Depends(auth)])
def job_video(jid: str):
    j = JOBS.get(jid)
    if not j or not j["video"]:
        raise HTTPException(status_code=404, detail="تیار نہیں")
    return FileResponse(j["video"], media_type="video/mp4", filename="AlAnsar.mp4")
