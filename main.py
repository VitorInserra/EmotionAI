import os
import asyncio
from sqlalchemy.orm import Session
import uvicorn
import pandas as pd
import time
import threading
import uuid
import EpocX
from db_handling import EpocXData as EXD
from db_handling.VRData import VRData
from fastapi import FastAPI, Depends


global_session_id: str = None


def set_global_session_id():
    """Initialize the global session ID."""
    global global_session_id
    global_session_id = str(uuid.uuid4())
    print(f"Initialized global session_id: {global_session_id}")


def get_global_session_id():
    """Retrieve the global session ID."""
    global global_session_id
    if not global_session_id:
        raise ValueError("Global session_id is not set.")
    return global_session_id


app = FastAPI()


@app.get("/")
async def set_session_id():
    start = time.time()
    set_global_session_id()
    t = threading.Thread(target=init_epoc_record)
    t.start()
    time.sleep(3)
    print(time.time() - start)
    return


def init_epoc_record():
    asyncio.run(EpocX.main())


@app.post("/datadump")
async def data_dump(data: VRData):
    session_id = get_global_session_id()
    start_stamp = data.start_stamp
    end_stamp = data.end_stamp
    eye_id = data.eye_id
    seen_words = data.seen_words
    text_version = data.text_version

    df = EpocX.pow_data_batch
    EXD.save_eeg_data(
        filename="datasets/curr_sesh.csv",
        user_id=0,
        session_id=session_id,
        start_stamp=start_stamp,
        end_stamp=end_stamp,
        eye_id=eye_id,
        text_version=text_version,
        seen_words=seen_words,
        arousal=-1,
        valence=-1,
        sensor_contact_quality=EpocX.sensor_contact_quality,
        df=df,
    )
    EpocX.pow_data_batch.drop(EpocX.pow_data_batch.index, inplace=True)

    return {"message": "VR data saved and recent EEG data recorded."}


def save_curr_sesh(path_a: str, path_b: str) -> pd.DataFrame:
    df_a = pd.read_csv(path_a)
    cols_a = df_a.columns.tolist()

    df_b = pd.read_csv(path_b, usecols=cols_a)

    # Concatenate row-wise
    combined = pd.concat([df_a, df_b[cols_a]], ignore_index=True)

    combined.to_csv(path_a, index=False)
    return combined


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8083, reload=False)
    while True:
        save_session_recordings = input("Do you want to save session recordings? [Y/N] ")
        if save_session_recordings.lower() == "y":
            save_curr_sesh("datasets/DECOD.csv", "datasets/curr_sesh.csv")
            break
        elif save_session_recordings.lower() == "n":
            sure = input("Are you sure?[Y/N] ")
            if sure.lower() == "y":
                break