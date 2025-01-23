# Databricks notebook source
# MAGIC %md
# MAGIC # Pipeline para Procesamiento e Ingeniería de Características para Detección de Fraude en Transacciones de Tarjetas de Crédito
# MAGIC
# MAGIC Este notebook implementa un pipeline diseñado para una **institución financiera** que busca construir un sistema de detección de fraudes para identificar transacciones sospechosas con tarjetas de crédito. La solución está optimizada para ejecutarse en **Databricks Community Edition**, aprovechando las capacidades de machine learning (ML) fundamentales mientras se trabaja dentro de las limitaciones de la plataforma.
# MAGIC
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Propósito General del Notebook
# MAGIC 1. **Cargar datos crudos:** Importa datos desde un archivo CSV que contiene información de transacciones de tarjetas de crédito.
# MAGIC 2. **Ingeniería de características:** Realiza las siguientes transformaciones clave:
# MAGIC    - **Velocidad de transacciones:** Calcula el número de transacciones realizadas por cliente en una ventana temporal definida.
# MAGIC    - **Velocidad de montos:** Calcula el monto total gastado por cliente en la misma ventana temporal.
# MAGIC    - **Perfilado de categoría de comerciante:** Analiza las transacciones realizadas en cada categoría comercial para identificar patrones de gasto.
# MAGIC    - **Patrones temporales:** Extrae características como la hora del día y el día de la semana para detectar comportamientos o anomalías.
# MAGIC 3. **Preparación para modelado:** Genera un conjunto de datos enriquecido y listo para entrenar modelos de aprendizaje automático.
# MAGIC 4. **Almacenamiento en formato Delta:** Guarda los datos procesados en formato **Delta**, optimizado para consultas y análisis en herramientas como Databricks.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Rutas Necesarias para la Configuración
# MAGIC
# MAGIC * **Nota:** Las dependecias necesarias para ejecutarlo están el la primera celda 
# MAGIC
# MAGIC ### **Archivos de Entrada**
# MAGIC - Los datos originales deben subirse en formato CSV (ejemplo: `credit_card_transactions.csv`).
# MAGIC - Sube estos archivos a la ruta predeterminada: `/FileStore/tables/` en Databricks.
# MAGIC - Ejemplo:
# MAGIC
# MAGIC /FileStore/tables/credit_card_transactions.csv
# MAGIC
# MAGIC
# MAGIC ### **Archivos de Salida**
# MAGIC - Los resultados procesados del pipeline se almacenarán en las siguientes rutas:
# MAGIC - **Conjunto de entrenamiento:** `/FileStore/tables/credit_card_transactions_dirt_to_train.csv`.
# MAGIC - **Conjunto de inferencia:** `/FileStore/tables/credit_card_transactions_dirt_to_inf.csv`.
# MAGIC - **Datos enriquecidos en formato Delta:**
# MAGIC   ```python
# MAGIC   output_path = "/FileStore/tables/output_delta_table_datapipe_feature_eng_to_train"
# MAGIC   save_to_delta(df, output_path)
# MAGIC   ```
# MAGIC   Este archivo contiene los datos enriquecidos listos para entrenamiento y su almacenamiento en formato Delta asegura compatibilidad y eficiencia para producción.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Ejecución Completa del Pipeline
# MAGIC
# MAGIC ### 1. **Cargar Datos Originales**
# MAGIC  - Los datos crudos se cargan desde el archivo CSV ubicado en `/FileStore/tables/credit_card_transactions.csv`.
# MAGIC
# MAGIC ### 2. **Dividir los Datos en Conjuntos**
# MAGIC  - **80%** para entrenamiento.
# MAGIC  - **20%** para inferencia.  
# MAGIC  - Estos conjuntos se guardan como CSV en las rutas mencionadas.
# MAGIC
# MAGIC ### 3. **Aplicar Ingeniería de Características**
# MAGIC  - **`transaction_velocity`:** Calcula la velocidad de transacciones.
# MAGIC  - **`amount_velocity`:** Calcula la velocidad de montos.
# MAGIC  - **`merchant_category_profiling`:** Perfilado de categorías comerciales.
# MAGIC  - **`time_based_patterns`:** Extrae patrones temporales.
# MAGIC
# MAGIC ### 4. **Guardar Datos Procesados**
# MAGIC  - Los datos enriquecidos se guardan en formato Delta para modelado:
# MAGIC    ```
# MAGIC    /FileStore/tables/output_delta_table_datapipe_feature_eng_to_train
# MAGIC    ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Contenido del Notebook
# MAGIC
# MAGIC ### **1. Documentación Detallada**
# MAGIC  - Explicaciones de las transformaciones y métodos aplicados.
# MAGIC  - Análisis exploratorio inicial de los datos.
# MAGIC
# MAGIC ### **2. Experimentos Relevantes**
# MAGIC  - Incluye código y pasos intermedios utilizados durante el desarrollo del pipeline, como pruebas de funciones y validaciones.
# MAGIC
# MAGIC ### **3. Código Listo para Producción**
# MAGIC  - Todas las transformaciones están encapsuladas en funciones reutilizables.
# MAGIC  - El código final puede trasladarse fácilmente a un **script** o **pipeline de producción** donde sea necesario.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Pasos para Producción
# MAGIC
# MAGIC 1. **Mover el Código a un Script de Producción**
# MAGIC  - Extrae las funciones y lógica principales del notebook para integrarlas en un script Python independiente o en un sistema ETL automatizado.
# MAGIC
# MAGIC 2. **Configurar Rutas en el Entorno de Producción**
# MAGIC  - Asegúrate de que las rutas de entrada y salida estén configuradas correctamente para el entorno donde se ejecutará.
# MAGIC
# MAGIC 3. **Ejecutar en el Entorno Final**
# MAGIC  - El pipeline está diseñado para trabajar en Databricks, pero puede adaptarse para ejecutarse localmente o en otro entorno Spark.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Notas Adicionales
# MAGIC Este notebook combina documentación, experimentos y un pipeline final listo para su implementación en un entorno de producción o investigación avanzada.
# MAGIC
# MAGIC

# COMMAND ----------

!pip install imbalanced-learn
!pip install mlflow
!pip install pytest pytest-cov
!pip install pytest-ipynb 

#%fs ls /FileStore/tables/
#dbutils.fs.rm("/FileStore/tables/engineered_features.delta", True) 

# COMMAND ----------

# MAGIC %md
# MAGIC # Initial sampling generation for training and inference 
# MAGIC
# MAGIC * Training (80%): 800,000 records.
# MAGIC   * Model building and tuning.
# MAGIC * Inference (20%): 200,000 records.
# MAGIC   * Data to test the model in production.

# COMMAND ----------

from pyspark.sql import SparkSession

# Crear sesión de Spark
spark = SparkSession.builder.appName("SplitDataset").getOrCreate()

# Ruta del archivo original
file_path = "/FileStore/tables/credit_card_transactions.csv"

# Cargar el DataFrame original
df_muestreo = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(file_path)

# Dividir el DataFrame en 80% para entrenamiento y 20% para inferencia
train_df, inf_df = df_muestreo.randomSplit([0.8, 0.2], seed=42)

# Guardar el DataFrame de entrenamiento
train_path = "/FileStore/tables/credit_card_transactions_dirt_to_train.csv"
train_df.write.format("csv").option("header", "true").mode("overwrite").save(train_path)

# Guardar el DataFrame de inferencia
inf_path = "/FileStore/tables/credit_card_transactions_dirt_to_inf.csv"
inf_df.write.format("csv").option("header", "true").mode("overwrite").save(inf_path)

# Confirmación
print(f"Archivos guardados: \n- Entrenamiento: {train_path} \n- Inferencia: {inf_path}")

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

## Solo para ver la salida de la ejecución

if __name__ == "__main__":
    try:
        # Configuración inicial
        logger.info("Starting the data pipeline.")
        spark = create_spark_session()

        # Ruta del archivo
        file_path = "/FileStore/tables/credit_card_transactions_dirt_to_train.csv"  
        
        # Cargar datos desde el archivo
        df = load_data(spark, file_path)

        # Mostrar el DataFrame original
        print("DataFrame Original:")
        df.show(20, truncate=False)

    except Exception as e:
        logger.error("Pipeline execution failed: %s", str(e))
        raise


# COMMAND ----------

df.schema

# COMMAND ----------

df.describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Análisis de Columnas del DataFrame Original
# MAGIC
# MAGIC #### 1. `transaction_id`
# MAGIC - **Conteo:** 800,218 registros.
# MAGIC - **Mínimo y Máximo:** IDs únicos desde `"00000a530069"` hasta `"fffff28ca038"`.
# MAGIC - **Conclusión:** No hay valores faltantes en esta columna, y parece ser un identificador único para cada transacción.
# MAGIC
# MAGIC #### 2. `customer_id`
# MAGIC - **Conteo:** 800,218 registros.
# MAGIC - **Rango de IDs:** Desde `"CUST_00000000"` hasta `"CUST_00051010"`.
# MAGIC - **Conclusión:** Todos los registros tienen un cliente asociado, lo que permite agrupar transacciones por cliente para análisis detallado.
# MAGIC
# MAGIC #### 3. `amount`
# MAGIC - **Media:** 123.13 (monto promedio de las transacciones).
# MAGIC - **Desviación Estándar:** 109.63 (indica una variabilidad moderada en los montos).
# MAGIC - **Mínimo y Máximo:** Transacciones desde 1.00 hasta 3,999.99.
# MAGIC - **Conclusión:** Hay una gran dispersión en los montos de transacciones, lo que puede reflejar distintos tipos de compras, desde pequeños gastos hasta compras más grandes.
# MAGIC
# MAGIC #### 4. `merchant_category`
# MAGIC - **Conteo:** 800,218 registros.
# MAGIC - **Valores Ejemplo:** Incluye categorías como `"digital_goods"`, `"travel"`.
# MAGIC - **Conclusión:** Todas las transacciones están asociadas a una categoría comercial, lo que permite analizar el comportamiento de gasto por categoría.
# MAGIC
# MAGIC #### 5. `merchant_country`
# MAGIC - **Rango de Países:** Desde `"CA"` hasta `"ZZ"`.
# MAGIC - **Conclusión:** La columna incluye transacciones internacionales, y `"ZZ"` podría indicar un valor anómalo o de datos no clasificados.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transaction Velocity
# MAGIC
# MAGIC ### **Propósito:**
# MAGIC Calcula el número de transacciones realizadas por cada cliente en los últimos 7 días desde el momento de cada transacción.
# MAGIC
# MAGIC ### **Proceso:**
# MAGIC 1. Convierte la columna `timestamp` a segundos (`timestamp_seconds`) para facilitar cálculos temporales.
# MAGIC 2. Define una ventana temporal de 7 días agrupada por `customer_id` y ordenada por tiempo.
# MAGIC 3. Calcula la cantidad de transacciones en esa ventana y agrega una nueva columna llamada `transaction_velocity`.
# MAGIC
# MAGIC ### **Output:**
# MAGIC Un DataFrame que incluye:
# MAGIC - **`timestamp_seconds`**: La fecha de cada transacción en segundos desde Unix Epoch.
# MAGIC - **`transaction_velocity`**: El número de transacciones realizadas por cliente en los últimos 7 días.
# MAGIC

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

if __name__ == "__main__":
    try:
        # Configuración inicial
        logger.info("Starting the data pipeline.")
        spark = create_spark_session()

        # Ruta del archivo
        file_path = "/FileStore/tables/credit_card_transactions_dirt_to_train.csv"  
        
        # Cargar datos desde el archivo
        df = load_data(spark, file_path)

        # Aplicar la función transaction_velocity
        df_with_velocity = transaction_velocity(df)

        # Mostrar el DataFrame con la velocidad de transacciones
        print("DataFrame con Velocidad de Transacciones:")
        df_with_velocity.show(20, truncate=False)

    except Exception as e:
        logger.error("Pipeline execution failed: %s", str(e))
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Amount Velocity
# MAGIC
# MAGIC ### **Propósito:**
# MAGIC Calcula el monto total de las transacciones realizadas por cada cliente en los últimos 7 días desde el momento de cada transacción.
# MAGIC
# MAGIC ### **Proceso:**
# MAGIC 1. Define una ventana temporal de 7 días agrupada por `customer_id` y ordenada por `timestamp_seconds`.
# MAGIC 2. Suma los valores de la columna `amount` dentro de la ventana para cada cliente.
# MAGIC 3. Agrega una nueva columna llamada `amount_velocity` que contiene el monto total calculado.
# MAGIC
# MAGIC ### **Output:**
# MAGIC Un DataFrame que incluye:
# MAGIC - **`amount_velocity`**: El monto total de las transacciones realizadas por cliente en los últimos 7 días.
# MAGIC

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

if __name__ == "__main__":
    try:
        # Configuración inicial
        logger.info("Starting the data pipeline.")
        spark = create_spark_session()

        # Ruta del archivo
        file_path = "/FileStore/tables/credit_card_transactions_dirt_to_train.csv"
        
        # Cargar datos desde el archivo
        df = load_data(spark, file_path)

        # Aplicar la función transaction_velocity
        df_with_velocity = transaction_velocity(df)

        # Mostrar el DataFrame con la velocidad de transacciones
        print("DataFrame con Velocidad de Transacciones:")
        df_with_velocity.show(20, truncate=False)

        # Aplicar la función amount_velocity sobre df_with_velocity
        df_with_amount_velocity = amount_velocity(df_with_velocity)

        # Mostrar el DataFrame con la velocidad de montos
        print("DataFrame con Velocidad de Montos (Amount Velocity):")
        df_with_amount_velocity.show(20, truncate=False)

    except Exception as e:
        logger.error("Pipeline execution failed: %s", str(e))
        raise


# COMMAND ----------

# MAGIC %md
# MAGIC ## Merchant Category Profiling
# MAGIC
# MAGIC ### **Propósito:**
# MAGIC Calcula el número de transacciones realizadas por cada cliente en una categoría comercial específica.
# MAGIC
# MAGIC ### **Proceso:**
# MAGIC 1. Define una ventana agrupada por `customer_id` y `merchant_category`.
# MAGIC 2. Cuenta el número de transacciones (`transaction_id`) dentro de cada categoría para cada cliente.
# MAGIC 3. Agrega una nueva columna llamada `merchant_category_count` que contiene el número total de transacciones por categoría comercial.
# MAGIC
# MAGIC ### **Output:**
# MAGIC Un DataFrame que incluye:
# MAGIC - **`merchant_category_count`**: El número de transacciones realizadas por cliente en cada categoría comercial.
# MAGIC

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

if __name__ == "__main__":
    try:
        # Configuración inicial
        logger.info("Starting the data pipeline.")
        spark = create_spark_session()

        # Ruta del archivo
        file_path = "/FileStore/tables/credit_card_transactions_dirt_to_train.csv"
        
        # Cargar datos desde el archivo
        df = load_data(spark, file_path)

        # Aplicar la función transaction_velocity
        df_with_velocity = transaction_velocity(df)

        # Aplicar la función amount_velocity sobre df_with_velocity
        df_with_amount_velocity = amount_velocity(df_with_velocity)

        # Aplicar la función merchant_category_profiling
        df_with_category_profiling = merchant_category_profiling(df_with_amount_velocity)

        # Mostrar el DataFrame con el perfil de categorías comerciales
        print("DataFrame con Perfil de Categorías Comerciales (Merchant Category Profiling):")
        df_with_category_profiling.show(20, truncate=False)

    except Exception as e:
        logger.error("Pipeline execution failed: %s", str(e))
        raise


# COMMAND ----------

# MAGIC %md
# MAGIC ## Time-Based Patterns
# MAGIC
# MAGIC ### **Propósito:**
# MAGIC Extrae patrones temporales de las transacciones, como la hora del día y el día de la semana, para análisis de comportamiento.
# MAGIC
# MAGIC ### **Proceso:**
# MAGIC 1. Calcula la hora del día a partir de la columna `timestamp` y la agrega como una nueva columna llamada `hour_of_day`.
# MAGIC 2. Determina el día de la semana a partir de la columna `timestamp` y lo agrega como una nueva columna llamada `day_of_week`.
# MAGIC
# MAGIC ### **Output:**
# MAGIC Un DataFrame que incluye:
# MAGIC - **`hour_of_day`**: La hora del día en que se realizó la transacción.
# MAGIC - **`day_of_week`**: El día de la semana en que ocurrió la transacción.
# MAGIC

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

if __name__ == "__main__":
    try:
        # Configuración inicial
        logger.info("Starting the data pipeline.")
        spark = create_spark_session()

        # Ruta del archivo
        file_path = "/FileStore/tables/credit_card_transactions_dirt_to_train.csv"
        
        # Cargar datos desde el archivo
        df = load_data(spark, file_path)

        # Aplicar la función transaction_velocity
        df_with_velocity = transaction_velocity(df)

        # Aplicar la función amount_velocity sobre df_with_velocity
        df_with_amount_velocity = amount_velocity(df_with_velocity)

        # Aplicar la función merchant_category_profiling
        df_with_category_profiling = merchant_category_profiling(df_with_amount_velocity)

        # Aplicar la función time_based_patterns
        df_with_time_patterns = time_based_patterns(df_with_category_profiling)

        # Mostrar el DataFrame con patrones basados en tiempo
        print("DataFrame con Patrones Basados en Tiempo (Time-Based Patterns):")
        df_with_time_patterns.show(20, truncate=False)

    except Exception as e:
        logger.error("Pipeline execution failed: %s", str(e))
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
        file_path = "/FileStore/tables/credit_card_transactions_dirt_to_train.csv"
        df = load_data(spark, file_path)

        # Preprocess data
        df = preprocess_data(df)

        # Perform feature engineering
        df = full_feature_engineering(df)

        # Save the processed data
        output_path = "/FileStore/tables/output_delta_table_datapipe_feature_eng_to_train"
        save_to_delta(df, output_path)

        logger.info("Data pipeline completed successfully.")
    except Exception as e:
        logger.error("Pipeline execution failed: %s", str(e))
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC # Pruebas Unitarias para Funciones de Procesamiento
# MAGIC
# MAGIC ## 1. `test_preprocess_data`
# MAGIC ### **Propósito:**
# MAGIC Verificar que `preprocess_data`:
# MAGIC - Convierte `amount` a `double` y `timestamp` a `timestamp`.
# MAGIC - Conserva las columnas necesarias.
# MAGIC
# MAGIC ### **Validación:**
# MAGIC La prueba pasa si las columnas `amount` y `timestamp` están presentes en el DataFrame procesado.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 2. `test_transaction_velocity`
# MAGIC ### **Propósito:**
# MAGIC Confirmar que `transaction_velocity`:
# MAGIC - Calcula correctamente la velocidad de transacciones en una ventana de 7 días.
# MAGIC - Agrega la columna `transaction_velocity`.
# MAGIC
# MAGIC ### **Validación:**
# MAGIC La prueba pasa si el DataFrame resultante contiene la columna `transaction_velocity`.
# MAGIC
# MAGIC **Nota:** Estas pruebas unitarias, debido a limitaciones de tiempo y a la imposibilidad de ejecutarlas directamente en el mismo notebook, se dejaron de forma hipotética (no las ejecuté como tal, pero aún asi las hice por el requerimiento). Según la metodología de pytest, estas pruebas se ejecutan en archivos `.py`. Por esta razón, decidí priorizar otros pasos del proceso.
# MAGIC

# COMMAND ----------

import pytest
from pyspark.sql import SparkSession

# Crear una sesión de Spark para pruebas
@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder \
        .appName("PytestSparkSession") \
        .master("local[*]") \
        .getOrCreate()

# Crear un DataFrame de prueba
@pytest.fixture
def sample_data(spark):
    data = [
        ("T1", "C1", "2025-01-01 10:00:00", 100.0, "retail", "US", True),
        ("T2", "C1", "2025-01-02 11:00:00", 150.0, "retail", "US", False),
    ]
    schema = ["transaction_id", "customer_id", "timestamp", "amount", "merchant_category", "merchant_country", "card_present"]
    return spark.createDataFrame(data, schema)

# Definir las pruebas
def test_preprocess_data(sample_data):
    df = preprocess_data(sample_data)
    assert "amount" in df.columns
    assert "timestamp" in df.columns

def test_transaction_velocity(sample_data):
    df = preprocess_data(sample_data)
    df = transaction_velocity(df)
    assert "transaction_velocity" in df.columns

# COMMAND ----------

#!pytest --maxfail=5 --disable-warnings


# COMMAND ----------

#!pytest --cov=. --cov-report=term-missing

