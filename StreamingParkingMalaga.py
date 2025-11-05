import os
import time
import requests
import shutil
import threading
from datetime import datetime, timedelta

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# HADOOP / WINUTILS CONFIGURATION
HADOOP_HOME = r"C:/hadoop/hadoop-3.2.2"
WINUTILS_BIN = r"C:/hadoop/hadoop-3.2.2/bin"

# Set environment variables
os.environ["HADOOP_HOME"] = HADOOP_HOME
os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + WINUTILS_BIN
os.environ["JAVA_TOOL_OPTIONS"] = "-Dorg.apache.hadoop.io.nativeio.NativeIO.disable=true"

# Allow Python (3.8+) to load Hadoop DLLs
try:
    os.add_dll_directory(WINUTILS_BIN)
except Exception:
    pass


# STREAMING CONFIGURATION
DATA_URL = "https://datosabiertos.malaga.eu/recursos/aparcamientos/ocupappublicosmun/ocupappublicosmun.csv"
BASE_DIR = os.path.abspath(os.getcwd())
INPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "data", "input")).replace("\\", "/")
PERIOD = 60  # download interval in seconds

# Official CSV schema
CSV_SCHEMA = StructType([
    StructField("dato", StringType(), True),
    StructField("id", StringType(), True),
    StructField("libres", IntegerType(), True),
])

# Temporary state for variation analysis
last_updates = {}
history = []


# DOWNLOADER FUNCTION
def downloader():
    """Periodically downloads the CSV into INPUT_DIR."""
    os.makedirs(INPUT_DIR, exist_ok=True)
    while True:
        try:
            resp = requests.get(DATA_URL, timeout=10)
            if resp.status_code == 200:
                tmp_path = os.path.join(INPUT_DIR, f"tmp_parking_{int(time.time())}.csv")
                with open(tmp_path, "wb") as f:
                    f.write(resp.content)

                final_path = os.path.join(INPUT_DIR, f"parking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                shutil.move(tmp_path, final_path)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Downloaded: {final_path}")
            else:
                print(f"HTTP Error {resp.status_code}")
        except Exception as e:
            print(f"Download error: {e}")

        time.sleep(PERIOD)


# OPTIONAL: LINE PLOT
def plot_history():
    """Displays the evolution of free parking spots over the last 5 minutes."""
    if not history:
        return

    try:
        df = pd.DataFrame(history, columns=["timestamp", "id", "libres"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        plt.figure(figsize=(8, 4))
        for pid in sorted(df["id"].unique()):
            sub = df[df["id"] == pid]
            plt.plot(sub["timestamp"], sub["libres"], marker="o", linewidth=1.2, label=pid)

        plt.title("Evolution of free parking spaces (last 5 minutes)")
        plt.xlabel("Time")
        plt.ylabel("Free spaces")

        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
        plt.close()

    except Exception as e:
        print(f"Plot error: {e}")


# OPTIONAL: BATCH ANALYSIS
def analyze_batch(df, batch_id):
    """Micro-batch analysis for changes and visualization."""
    global last_updates, history

    pdf = df.toPandas()
    if pdf.empty:
        return

    now = datetime.now()
    changed = []

    for _, row in pdf.iterrows():
        pid = row.get("id")
        libres = row.get("libres")
        if pid is None or libres is None:
            continue

        prev = last_updates.get(pid)

        if not prev or prev[0] != libres:
            changed.append((pid, libres))
            last_updates[pid] = (libres, now)
            history.append((now, pid, libres))

    # Keep only the last 5 minutes of history
    cutoff = now - timedelta(minutes=5)
    history = [x for x in history if x[0] > cutoff]

    if changed:
        print("Parking lots with changes in the last 5 minutes:")
        for pid, libres in changed:
            print(f"   • {pid}: {libres} free spots")
    else:
        print("No changes in the last 5 minutes.")

    plot_history()


# MAIN
def main():
    print("Starting StreamingParkingMalaga...")
    os.makedirs(INPUT_DIR, exist_ok=True)

    # Start downloader in background thread
    threading.Thread(target=downloader, daemon=True).start()

    # Extra Java options for Spark (fix for NativeIO on Windows)
    extra_java = (
        f"-Dhadoop.home.dir={HADOOP_HOME} "
        f"-Djava.library.path={WINUTILS_BIN} "
        f"-Dorg.apache.hadoop.io.nativeio.NativeIO.disable=true"
    )

    # Spark session for Windows
    spark = (
        SparkSession.builder
        .appName("StreamingParkingMalaga")
        .master("local[*]")
        .config("spark.hadoop.io.nativeio.NativeIO.disable", "true")
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem")
        .config("spark.hadoop.fs.AbstractFileSystem.file.impl", "org.apache.hadoop.fs.local.LocalFs")
        .config("spark.driver.extraJavaOptions", extra_java)
        .config("spark.executor.extraJavaOptions", extra_java)
        .config("spark.sql.streaming.schemaInference", "true")
        .getOrCreate()
    )

    # JVM debug
    jvm = spark._jvm
    print("hadoop.home.dir (JVM) =", jvm.java.lang.System.getProperty("hadoop.home.dir"))
    print("java.library.path (JVM) =", jvm.java.lang.System.getProperty("java.library.path"))
    print(f"Monitoring folder: {INPUT_DIR}")

    # STREAM READING
    streamDF = (
        spark.readStream
        .option("header", True)
        .schema(CSV_SCHEMA)
        .csv(INPUT_DIR)
        .select("id", "libres")
    )

    # Console output stream
    query_console = (
        streamDF.writeStream
        .format("console")
        .outputMode("append")
        .option("truncate", "false")
        .trigger(processingTime=f"{PERIOD} seconds")
        .start()
    )

    # Optional analysis stream
    query_analysis = (
        streamDF.writeStream
        .foreachBatch(analyze_batch)
        .outputMode("append")
        .trigger(processingTime=f"{PERIOD} seconds")
        .start()
    )

    try:
        query_console.awaitTermination()
        query_analysis.awaitTermination()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            query_console.stop()
        except Exception:
            pass
        try:
            query_analysis.stop()
        except Exception:
            pass
        spark.stop()


# ENTRY POINT
if __name__ == "__main__":
    main()
