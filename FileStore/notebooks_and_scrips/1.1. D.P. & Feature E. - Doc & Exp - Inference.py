# Databricks notebook source
# MAGIC %md
# MAGIC # Pipeline para Ingeniería de Características para Inferencia en Transacciones de Tarjetas de Crédito
# MAGIC
# MAGIC Este notebook implementa un pipeline de **ingeniería de características** diseñado para preparar un conjunto de datos de transacciones de tarjetas de crédito exclusivamente para tareas de **inferencia**. Está optimizado para ejecutarse en **Databricks Community Edition**, maximizando las capacidades de procesamiento de datos dentro de las limitaciones de la plataforma.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Propósito General del Notebook
# MAGIC
# MAGIC 1. **Cargar datos crudos:** Importa datos desde un archivo CSV con información de transacciones de tarjetas de crédito.
# MAGIC 2. **Ingeniería de características:** Realiza transformaciones clave para preparar los datos:
# MAGIC    - **Velocidad de transacciones:** Número de transacciones realizadas por cliente en una ventana temporal definida.
# MAGIC    - **Velocidad de montos:** Monto total gastado por cliente en la misma ventana temporal.
# MAGIC    - **Perfilado de categoría de comerciante:** Análisis de patrones de gasto por categoría comercial.
# MAGIC    - **Patrones temporales:** Extracción de características como la hora del día y el día de la semana.
# MAGIC 3. **Preparación para inferencia:** Genera un conjunto de datos enriquecido listo para ser usado por modelos previamente entrenados.
# MAGIC 4. **Almacenamiento en formato Delta:** Guarda los datos procesados en formato **Delta**, optimizado para consultas y análisis.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Rutas Necesarias para la Configuración
# MAGIC
# MAGIC ### **Dependencias**
# MAGIC - Todas las dependencias necesarias están instaladas en la primera celda del notebook.
# MAGIC
# MAGIC ### **Archivos de Entrada**
# MAGIC - Los datos ya están en CSV por la ejecución del notebook anterior (`credit_card_transactions_dirt_to_inf.csv`).
# MAGIC - Ubicación en Databricks: `/FileStore/tables/`.
# MAGIC
# MAGIC ### **Archivos de Salida**
# MAGIC - **Datos procesados en formato Delta:** Los datos procesados se encontraran en `/FileStore/tables/output_delta_table_datapipe_feature_eng_to_inf` una vez finalizada la ejecución del notebook 
# MAGIC
# MAGIC **Nota:** La escases de doc, unit tests y experimentos en este notebook es porque es igual al notebook anterior solo que cambiando el archivo de salida y entrada. 
# MAGIC

# COMMAND ----------

!pip install imbalanced-learn
!pip install mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Pipeline and Feature Engineering
# MAGIC
# MAGIC * ### Load and Process the Provided Credit Card Transaction Dataset 

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum, unix_timestamp, hour, dayofweek
from pyspark.sql.window import Window
import logging

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def create_spark_session(app_name="CreditCardFraudDetection"):
    """Create and return a Spark session."""
    try:
        spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .getOrCreate()
        logger.info("Spark session created successfully.")
        return spark
    except Exception as e:
        logger.error("Failed to create Spark session: %s", str(e))
        raise

def load_data(spark, file_path):
    """Load data from the provided file path."""
    try:
        df = spark.read.format("csv") \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .load(file_path)
        logger.info("Data loaded successfully from %s.", file_path)
        return df
    except Exception as e:
        logger.error("Failed to load data: %s", str(e))
        raise

def preprocess_data(df):
    """Initial preprocessing: Cast columns to appropriate data types."""
    try:
        df = df.withColumn("amount", col("amount").cast("double"))
        df = df.withColumn("timestamp", col("timestamp").cast("timestamp"))
        logger.info("Data preprocessing completed.")
        return df
    except Exception as e:
        logger.error("Error during preprocessing: %s", str(e))
        raise


# COMMAND ----------

# MAGIC %md
# MAGIC ### Transaction Velocity

# COMMAND ----------

def transaction_velocity(df):
    """Calculate transaction velocity (number of transactions per time window)."""
    try:
        df = df.withColumn("timestamp_seconds", unix_timestamp(col("timestamp")))
        time_window = Window.partitionBy("customer_id").orderBy("timestamp_seconds").rangeBetween(-604800, 0)
        df = df.withColumn("transaction_velocity", count("transaction_id").over(time_window))
        logger.info("Transaction velocity calculated.")
        return df
    except Exception as e:
        logger.error("Error calculating transaction velocity: %s", str(e))
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ### Amount Velocity

# COMMAND ----------

def amount_velocity(df):
    """Calculate amount velocity (total amount per time window)."""
    try:
        time_window = Window.partitionBy("customer_id").orderBy("timestamp_seconds").rangeBetween(-604800, 0)
        df = df.withColumn("amount_velocity", sum("amount").over(time_window))
        logger.info("Amount velocity calculated.")
        return df
    except Exception as e:
        logger.error("Error calculating amount velocity: %s", str(e))
        raise


# COMMAND ----------

# MAGIC %md
# MAGIC ### Merchant Category Profiling

# COMMAND ----------

def merchant_category_profiling(df):
    """Calculate merchant category profiling for each customer."""
    try:
        category_window = Window.partitionBy("customer_id", "merchant_category")
        df = df.withColumn("merchant_category_count", count("transaction_id").over(category_window))
        logger.info("Merchant category profiling completed.")
        return df
    except Exception as e:
        logger.error("Error during merchant category profiling: %s", str(e))
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ### Time-Based Patterns 

# COMMAND ----------

def time_based_patterns(df):
    """Extract time-based patterns like hour of day and day of week."""
    try:
        df = df.withColumn("hour_of_day", hour(col("timestamp")))
        df = df.withColumn("day_of_week", dayofweek(col("timestamp")))
        logger.info("Time-based patterns extracted.")
        return df
    except Exception as e:
        logger.error("Error extracting time-based patterns: %s", str(e))
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ### Implement Feature Engineering Using Spark SQL and Window Functions 

# COMMAND ----------

def full_feature_engineering(df):
    """Run all feature engineering steps sequentially."""
    try:
        df = transaction_velocity(df)
        df = amount_velocity(df)
        df = merchant_category_profiling(df)
        df = time_based_patterns(df)
        logger.info("All feature engineering steps completed.")
        return df
    except Exception as e:
        logger.error("Error during full feature engineering: %s", str(e))
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ### Store Engineered Features in Delta Format

# COMMAND ----------

def save_to_delta(df, output_path):
    """Save processed data to Delta format."""
    try:
        if not output_path:
            raise ValueError("Output path cannot be empty.")

        df.write.format("delta").mode("overwrite").save(output_path)
        logger.info("Data saved to Delta format at %s.", output_path)
    except Exception as e:
        logger.error("Failed to save data to Delta: %s", str(e))
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ### Production-Ready Code
# MAGIC * Main Function for Pipeline Execution

# COMMAND ----------

if __name__ == "__main__":
    try:
        # Setup
        logger.info("Starting the data pipeline.")
        spark = create_spark_session()

        # Load data
        file_path = "/FileStore/tables/credit_card_transactions_dirt_to_inf.csv"  # Replace with your file path
        df = load_data(spark, file_path)

        # Preprocess data
        df = preprocess_data(df)

        # Perform feature engineering
        df = full_feature_engineering(df)

        # Save the processed data
        output_path = "/FileStore/tables/output_delta_table_datapipe_feature_eng_to_inf"
        save_to_delta(df, output_path)

        logger.info("Data pipeline completed successfully.")
    except Exception as e:
        logger.error("Pipeline execution failed: %s", str(e))
        raise
