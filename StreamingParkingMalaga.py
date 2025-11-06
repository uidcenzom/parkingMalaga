import os
import time
import requests
import shutil
import threading
from datetime import datetime, timedelta

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

import pandas as pd
import plotly.graph_objects as go

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

# Windows environment setup
def setup_windows_environment():
    """Configures environment variables and Spark options for Windows"""
    if os.name != 'nt':
        print("Skipping Windows-specific setup (not on Windows).")
        return {}  # Return an empty dictionary if not on Windows

    print("Applying Windows/Hadoop environment setup...")

    # Define paths inside the function
    HADOOP_HOME = r"C:/hadoop/hadoop-3.2.2"
    WINUTILS_BIN = r"C:/hadoop/hadoop-3.2.2/bin"

    # Set environment variables
    os.environ["HADOOP_HOME"] = HADOOP_HOME
    os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + WINUTILS_BIN
    os.environ["JAVA_TOOL_OPTIONS"] = "-Dorg.apache.hadoop.io.nativeio.NativeIO.disable=true"

    try:
        os.add_dll_directory(WINUTILS_BIN)
    except Exception:
        print("Note: os.add_dll_directory not available (likely Python < 3.8).")
        pass

    extra_java = (
        f"-Dhadoop.home.dir={HADOOP_HOME} "
        f"-Djava.library.path={WINUTILS_BIN} "
        f"-Dorg.apache.hadoop.io.nativeio.NativeIO.disable=true"
    )

    return {
        "spark.hadoop.io.nativeio.NativeIO.disable": "true",
        "spark.hadoop.fs.file.impl": "org.apache.hadoop.fs.LocalFileSystem",
        "spark.hadoop.fs.AbstractFileSystem.file.impl": "org.apache.hadoop.fs.local.LocalFs",
        "spark.driver.extraJavaOptions": extra_java,
        "spark.executor.extraJavaOptions": extra_java
    }


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

def plot_history_plotty():
    """Displays the evolution of free parking spots over the last 5 minutes."""
    if not history:
        return

    try:
        df = pd.DataFrame(history, columns=["timestamp", "id", "libres"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        fig = go.Figure()

        for pid in sorted(df["id"].unique()):
            sub = df[df["id"] == pid]
            fig.add_trace(go.Scatter(
                x=sub["timestamp"],
                y=sub["libres"],
                mode='lines+markers',
                name=pid,
                line=dict(width=2),
                marker=dict(size=6)
            ))

        fig.update_layout(
            title="Evolution of free parking spaces (last 5 minutes)",
            xaxis_title="Time",
            yaxis_title="Free spaces",
            hovermode='x unified',
            template="plotly_white",
            height=500,
            showlegend=True,
            xaxis=dict(tickformat='%H:%M')  # Add tickformat here
        )

        # Save as interactive HTML file
        plot_path = os.path.join(BASE_DIR, "parking_plot_live.html")
        fig.write_html(plot_path, auto_open=False)
        print(f"Interactive plot saved to: {plot_path}")

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

        history.append((now, pid, libres))
        prev = last_updates.get(pid)

        if not prev or prev[0] != libres:
            changed.append((pid, libres))
            last_updates[pid] = (libres, now)

    # Keep only the last 5 minutes of history
    cutoff = now - timedelta(minutes=5)
    history = [x for x in history if x[0] > cutoff]

    if changed:
        print("Parking lots with changes in the last 5 minutes:")
        for pid, libres in changed:
            print(f"   • {pid}: {libres} free spots")
    else:
        print("No changes in the last 5 minutes.")
    plot_history_plotty()


def setup_data_folder():
    """Clean the input directory to prevent processing old files"""
    print(f"Cleaning input directory for a fresh start: {INPUT_DIR}")
    if os.path.exists(INPUT_DIR):
        try:
            shutil.rmtree(INPUT_DIR)
        except Exception as e:
            print(f"Warning: Could not clean directory. {e}")
    os.makedirs(INPUT_DIR, exist_ok=True)


def print_jvm_info(jvm):
    print("hadoop.home.dir (JVM) =", jvm.java.lang.System.getProperty("hadoop.home.dir"))
    print("java.library.path (JVM) =", jvm.java.lang.System.getProperty("java.library.path"))
    print(f"Monitoring folder: {INPUT_DIR}")


def createSparkSession():
    platform_configs = setup_windows_environment()
    builder = (
        SparkSession.builder
        .appName("StreamingParkingMalaga")
        .master("local[*]")
        .config("spark.sql.streaming.schemaInference", "true")
    )
    for key, value in platform_configs.items():
        builder = builder.config(key, value)

    return builder.getOrCreate()


# MAIN
def main():
    print("Starting StreamingParkingMalaga...")
    setup_data_folder()

    # Start downloader in background thread
    threading.Thread(target=downloader, daemon=True).start()

    spark = createSparkSession()

    print_jvm_info(spark._jvm)

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
        spark.streams.awaitAnyTermination()
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
