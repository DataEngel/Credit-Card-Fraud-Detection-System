# Databricks notebook source
# MAGIC %md
# MAGIC # Pipeline para Inferencia Batch en Detección de Fraude
# MAGIC
# MAGIC ## **Descripción**
# MAGIC Este pipeline está diseñado para aplicar un modelo preentrenado de detección de fraudes sobre un conjunto de datos transaccionales. Aprovecha PySpark y MLlib para cargar el modelo, generar predicciones y procesar los resultados de manera eficiente. El flujo se ejecuta en **Databricks Community Edition**, aprovechando su entorno escalable.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## **Objetivo**
# MAGIC Aplicar un modelo predictivo para clasificar transacciones de tarjetas de crédito como fraudulentas o no fraudulentas, priorizando aquellas de alto riesgo basándose en las probabilidades calculadas.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## **Configuraciones Requeridas**
# MAGIC ### **Rutas de Entrada y Salida**
# MAGIC - **Datos de entrada:** Tabla Delta con características procesadas en `/FileStore/tables/output_delta_table_datapipe_feature_eng_to_inf`.
# MAGIC - **Modelo preentrenado:** Ruta del modelo en `/dbfs/tmp/fraud_detection_cv_model`.
# MAGIC - **Resultados de inferencia:** CSV con transacciones clasificadas en `/FileStore/tables/scored_transactions_results_batch`.
# MAGIC - **Etiquetas verdaderas procesadas:** CSV con etiquetas convertidas en `/FileStore/tables/etiquetas_verdaderas_processed`.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## **Notas**
# MAGIC - Este pipeline está optimizado para manejar grandes volúmenes de datos mediante procesamiento distribuido.
# MAGIC - Las rutas configuradas deben ajustarse según el entorno en que se ejecute.
# MAGIC - Los resultados permiten priorizar transacciones para investigaciones adicionales y detección temprana de fraudes.
# MAGIC

# COMMAND ----------

!pip install mlflow
!pip install pytest

# COMMAND ----------

# MAGIC %md
# MAGIC # Just for explore

# COMMAND ----------

from pyspark.sql import SparkSession

# Crear una sesión de Spark
spark = SparkSession.builder \
    .appName("Read Delta Table") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

# Ruta del archivo Delta
delta_path = "/FileStore/tables/output_delta_table_datapipe_feature_eng_to_inf"

# Leer el archivo Delta en un DataFrame
df = spark.read.format("delta").load(delta_path)

# Mostrar las primeras filas del DataFrame
df.show(20, truncate=False)

# COMMAND ----------

df.describe().show()

# COMMAND ----------

df.printSchema

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Configuración Inicial: Logging y Sesión de Spark

# COMMAND ----------

import logging
from pyspark.sql import SparkSession

# Configuración del logger
def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)

logger = setup_logging()

# Crear una sesión de Spark
def create_spark_session(app_name="Batch Inference Pipeline"):
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


# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Model
# MAGIC
# MAGIC ### **Propósito:**
# MAGIC Cargar un modelo de regresión logística previamente entrenado desde una ruta especificada.
# MAGIC
# MAGIC ### **Proceso:**
# MAGIC 1. Se utiliza la función `LogisticRegressionModel.load()` para cargar el modelo desde la ubicación proporcionada.
# MAGIC 2. Se verifica que la carga sea exitosa y se registra en el logger.
# MAGIC
# MAGIC ### **Output:**
# MAGIC Un objeto `LogisticRegressionModel` listo para su uso en tareas de inferencia.

# COMMAND ----------

from pyspark.ml.classification import LogisticRegressionModel

# Cargar el modelo entrenado
def load_model(model_path):
    try:
        logger.info("Loading model from: %s", model_path)
        model = LogisticRegressionModel.load(model_path)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error("Failed to load model: %s", str(e))
        raise


# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data
# MAGIC
# MAGIC ### **Propósito:**
# MAGIC Cargar datos desde un archivo en formato Delta para realizar tareas de inferencia.
# MAGIC
# MAGIC ### **Proceso:**
# MAGIC 1. Se utiliza el método `spark.read.format("delta").load()` para leer los datos desde la ruta especificada.
# MAGIC 2. Se registra en el logger la cantidad de registros cargados para verificar la operación.
# MAGIC
# MAGIC ### **Output:**
# MAGIC Un DataFrame de PySpark con los datos cargados desde el archivo Delta, listo para procesar.
# MAGIC

# COMMAND ----------

# Cargar nuevos datos para inferencia
def load_data(spark, data_path):
    try:
        logger.info("Loading data from: %s", data_path)
        df = spark.read.format("delta").load(data_path)
        logger.info("Data loaded successfully. Record count: %d", df.count())
        return df
    except Exception as e:
        logger.error("Failed to load data: %s", str(e))
        raise


# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Features Column
# MAGIC
# MAGIC ### **Propósito:**
# MAGIC Generar una columna llamada `features` en el DataFrame que combine las columnas seleccionadas para ser utilizadas como entrada en un modelo de machine learning.
# MAGIC
# MAGIC ### **Proceso:**
# MAGIC 1. Se inicializa un `VectorAssembler` con las columnas de entrada especificadas en `feature_columns`.
# MAGIC 2. Se aplica el ensamblador al DataFrame, generando una nueva columna llamada `features` (o el nombre especificado en `output_column`).
# MAGIC 3. Se registra en el logger la creación exitosa de la columna.
# MAGIC
# MAGIC ### **Output:**
# MAGIC Un DataFrame de PySpark que incluye la columna adicional `features`, lista para ser utilizada en el entrenamiento o inferencia del modelo.
# MAGIC

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# Crear la columna 'features' en el dataset
def create_features_column(df, feature_columns, output_column="features"):
    try:
        logger.info("Creating features column...")
        assembler = VectorAssembler(inputCols=feature_columns, outputCol=output_column)
        df = assembler.transform(df)
        logger.info("Features column created successfully.")
        return df
    except Exception as e:
        logger.error("Error creating features column: %s", str(e))
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Batch Inference
# MAGIC
# MAGIC ### **Propósito:**
# MAGIC Realizar inferencia en lote utilizando un modelo previamente entrenado para generar predicciones sobre un conjunto de datos de entrada.
# MAGIC
# MAGIC ### **Proceso:**
# MAGIC 1. Se aplica el modelo a los datos de entrada para generar predicciones.
# MAGIC 2. Se seleccionan columnas clave del DataFrame de predicciones:
# MAGIC    - **`transaction_id`**: Identificador único de la transacción.
# MAGIC    - **`customer_id`**: Identificador del cliente asociado a la transacción.
# MAGIC    - **`features`**: Vector de características utilizado como entrada para el modelo.
# MAGIC    - **`probability`**: Vector que contiene las probabilidades de pertenencia a cada clase.
# MAGIC    - **`prediction`**: Clase predicha por el modelo (fraude o no fraude).
# MAGIC
# MAGIC ### **Output:**
# MAGIC Un DataFrame de PySpark que contiene:
# MAGIC - **`transaction_id`**
# MAGIC - **`customer_id`**
# MAGIC - **`features`**
# MAGIC - **`probability`**
# MAGIC - **`prediction`**
# MAGIC

# COMMAND ----------

# Realizar inferencia batch
def run_batch_inference(model, df):
    try:
        logger.info("Running batch inference...")
        predictions = model.transform(df)
        logger.info("Inference completed successfully.")
        return predictions.select(
            "transaction_id", "customer_id", "features", "probability", "prediction"
        )
    except Exception as e:
        logger.error("Error during batch inference: %s", str(e))
        raise



# COMMAND ----------

# MAGIC %md
# MAGIC # 7. Pipeline Principal
# MAGIC
# MAGIC Este es el flujo principal que llama a todas las funciones anteriores.

# COMMAND ----------

def main_pipeline():
    try:
        logger.info("Starting Batch Inference Pipeline...")

        # Configuraciones
        data_path = "/FileStore/tables/output_delta_table_datapipe_feature_eng_to_inf"
        model_path = "/dbfs/tmp/fraud_detection_cv_model"

        # Crear sesión de Spark
        spark = create_spark_session()

        # Cargar datos y modelo
        df = load_data(spark, data_path)
        model = load_model(model_path)

        # Crear la columna 'features' si es necesario
        feature_columns = [
            "transaction_velocity",
            "amount_velocity",
            "merchant_category_count",
            "hour_of_day",
            "day_of_week"
        ]
        df = create_features_column(df, feature_columns)

        # Realizar inferencia
        predictions = run_batch_inference(model, df)

        # Mostrar las primeras filas de las predicciones
        predictions.show(truncate=False)

        logger.info("Batch Inference Pipeline completed successfully.")
    except Exception as e:
        logger.error("Pipeline execution failed: %s", str(e))
        raise

if __name__ == "__main__":
    main_pipeline()

# COMMAND ----------

# MAGIC %md
# MAGIC # Guardar los resultados en un csv
# MAGIC
# MAGIC A consecuencia del los tipo de datos complejos, opté por pasarlo a un df de pandas, darle el formato apropiado para el analisis y luego guardarlo como un csv con pyspark 

# COMMAND ----------

# Contar los valores 0 y 1 en la columna 'prediction'
#predictions.groupBy("prediction").count().show()

# COMMAND ----------

logger.info("Starting Batch Inference Pipeline...")

# Configuraciones
data_path = "/FileStore/tables/output_delta_table_datapipe_feature_eng_to_inf"
model_path = "/dbfs/tmp/fraud_detection_cv_model"

# Crear sesión de Spark
spark = create_spark_session()

# Cargar datos y modelo
df = load_data(spark, data_path)
model = load_model(model_path)

# Crear la columna 'features' si es necesario
feature_columns = [
    "transaction_velocity",
    "amount_velocity",
    "merchant_category_count",
    "hour_of_day",
    "day_of_week"
]
df = create_features_column(df, feature_columns)

# Realizar inferencia
predictions = run_batch_inference(model, df)

# Mostrar las primeras filas de las predicciones
predictions.show(truncate=False)

# COMMAND ----------

predictions.printSchema

# COMMAND ----------

# Convertir el DataFrame PySpark a Pandas
predictions = predictions.toPandas()

# COMMAND ----------

# Separar probabilidades en columnas individuales
predictions['prob_0'] = predictions['probability'].apply(lambda x: x[0])  # Probabilidad de no fraude
predictions['prob_1'] = predictions['probability'].apply(lambda x: x[1])  # Probabilidad de fraude

# Seleccionar las columnas relevantes para el análisis
scored_transactions = predictions[['transaction_id', 'customer_id', 'prob_0', 'prob_1', 'prediction']]

# Ordenar por probabilidad de fraude (para priorizar transacciones de alto riesgo)
#scored_transactions = scored_transactions.sort_values(by='prob_1', ascending=False)

# Mostrar las primeras filas del DataFrame procesado
print(scored_transactions.head())

# COMMAND ----------

# Ver el esquema del DataFrame en Pandas
scored_transactions.info()

# COMMAND ----------

# 1. Convertir el DataFrame de Pandas a PySpark
scored_transactions_spark = spark.createDataFrame(scored_transactions)

# 2. Guardar el DataFrame en formato CSV en la ruta de Databricks
output_path = "/FileStore/tables/scored_transactions_results_batch"

scored_transactions_spark.write \
    .format("csv") \
    .mode("overwrite") \
    .option("header", "true") \
    .save(output_path)

print(f"Scored transactions saved successfully to {output_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Leer, formatear y guardas las etiquetas de los verdaderos positivos

# COMMAND ----------

# Ruta del archivo en formato Delta
file_path = "/FileStore/tables/output_delta_table_datapipe_feature_eng_to_inf"

# Leer el archivo en formato Delta
ground_truth = spark.read.format("delta").load(file_path)

# Seleccionar las columnas 'transaction_id' y 'is_fraud'
selected_columns = ground_truth.select("transaction_id", "is_fraud")

# Mostrar las primeras 10 filas
selected_columns.show(10, truncate=False)


# COMMAND ----------

from pyspark.sql.functions import when, col

# Convertir 'is_fraud' de true/false a 1.0/0.0
ground_truth = ground_truth.withColumn(
    "is_fraud",
    when(col("is_fraud") == "true", 1.0).when(col("is_fraud") == "false", 0.0).otherwise(None)
)

# Mostrar las primeras filas para verificar la conversión
ground_truth.select("transaction_id", "is_fraud").show(10, truncate=False)

# COMMAND ----------

# Guardar el DataFrame procesado en un archivo CSV
output_path = "/FileStore/tables/etiquetas_verdaderas_processed"

# Guardar el DataFrame con las etiquetas convertidas
ground_truth.select("transaction_id", "is_fraud") \
    .write \
    .format("csv") \
    .mode("overwrite") \
    .option("header", "true") \
    .save(output_path)

print(f"Processed ground truth saved successfully to {output_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Descripción de las Pruebas
# MAGIC
# MAGIC ### **`test_load_model`**
# MAGIC - **Propósito:**  
# MAGIC   Valida que la función puede cargar correctamente un modelo guardado en la ruta especificada.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **`test_load_data`**
# MAGIC - **Propósito:**  
# MAGIC   Comprueba que la función puede cargar datos desde una tabla Delta.
# MAGIC - **Validaciones:**  
# MAGIC   - El DataFrame devuelto tiene el esquema esperado.  
# MAGIC   - El número de registros coincide con el esperado.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **`test_create_features_column`**
# MAGIC - **Propósito:**  
# MAGIC   Asegura que la columna `features` se genere correctamente a partir de las columnas de entrada.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **`test_run_batch_inference`**
# MAGIC - **Propósito:**  
# MAGIC   Valida que la función puede ejecutar inferencias batch con un modelo de regresión logística.
# MAGIC - **Validaciones:**  
# MAGIC   - El DataFrame resultante contiene las columnas `prediction` y `probability`.
# MAGIC
# MAGIC
# MAGIC **Nota:** Estas pruebas unitarias, debido a limitaciones de tiempo y a la imposibilidad de ejecutarlas directamente en el mismo notebook, se dejaron de forma hipotética (no las ejecuté como tal, pero aún asi las hice por el requerimiento). Según la metodología de pytest, estas pruebas se ejecutan en archivos `.py`. Por esta razón, decidí priorizar otros pasos del proceso.

# COMMAND ----------

import pytest

@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder \
        .appName("UnitTest") \
        .master("local[*]") \
        .getOrCreate()

@pytest.fixture
def sample_data(spark):
    schema = StructType([
        StructField("transaction_id", StringType(), True),
        StructField("customer_id", StringType(), True),
        StructField("features", StringType(), True),
        StructField("is_fraud", DoubleType(), True)
    ])
    data = [
        ("T1", "C1", "[1.0, 2.0, 3.0]", 1.0),
        ("T2", "C2", "[2.0, 3.0, 4.0]", 0.0),
        ("T3", "C3", "[3.0, 4.0, 5.0]", 1.0),
    ]
    return spark.createDataFrame(data, schema)

def test_load_model(tmp_path):
    model_path = tmp_path / "model"
    model = LogisticRegressionModel(uid="logistic_regression")
    model.write().overwrite().save(str(model_path))
    loaded_model = load_model(str(model_path))
    assert isinstance(loaded_model, LogisticRegressionModel)

def test_load_data(spark, tmp_path):
    data_path = tmp_path / "data"
    data = [(1, "C1"), (2, "C2")]
    schema = StructType([
        StructField("transaction_id", StringType(), True),
        StructField("customer_id", StringType(), True)
    ])
    df = spark.createDataFrame(data, schema)
    df.write.format("delta").save(str(data_path))
    loaded_df = load_data(spark, str(data_path))
    assert isinstance(loaded_df, DataFrame)
    assert loaded_df.count() == 2

def test_create_features_column(sample_data):
    feature_columns = ["transaction_id", "customer_id"]
    df = create_features_column(sample_data, feature_columns)
    assert "features" in df.columns

def test_run_batch_inference(spark, sample_data):
    model = LogisticRegressionModel(uid="logistic_regression")
    model.setFeaturesCol("features").setLabelCol("is_fraud")
    model.fit(sample_data)
    predictions = run_batch_inference(model, sample_data)
    assert "prediction" in predictions.columns
    assert "probability" in predictions.columns
    assert predictions.count() == sample_data.count()

