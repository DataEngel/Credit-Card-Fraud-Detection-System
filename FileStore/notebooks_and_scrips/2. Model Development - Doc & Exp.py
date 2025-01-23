# Databricks notebook source
# MAGIC %md
# MAGIC # Modelo de Detección de Fraude en Tarjetas de Crédito
# MAGIC
# MAGIC Este notebook desarrolla un **modelo predictivo** para identificar transacciones fraudulentas, diseñado para ejecutarse en la **edición Community de Databricks**. Integra **MLflow** para el seguimiento de experimentos y aborda desafíos como el **desbalance de clases** y la evaluación eficiente del modelo, almacenando los resultados y modelos localmente debido a las limitaciones de la plataforma.
# MAGIC
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Configuraciones
# MAGIC
# MAGIC ### Datos de Entrada
# MAGIC - El conjunto de datos consiste en transacciones preprocesadas y enriquecidas con características ingenierizadas.
# MAGIC - **Formato de origen**: Tabla Delta.
# MAGIC - **Ruta**: `/FileStore/tables/output_delta_table_datapipe_feature_eng_to_train`.
# MAGIC
# MAGIC ### Datos de Salida
# MAGIC - **Ruta de guardado del modelo**: `/dbfs/tmp/fraud_detection_cv_model`.
# MAGIC   - El modelo entrenado se guarda localmente para su uso en procesos posteriores.
# MAGIC - **Seguimiento de experimentos**: `/Users/tu-correo@example.com/fraud_detection_experiment`.
# MAGIC   - Todos los parámetros, métricas y artefactos se rastrean mediante MLflow.
# MAGIC
# MAGIC ### Dependencias
# MAGIC - **MLflow**: Utilizado para el seguimiento de experimentos y almacenamiento de metadatos del modelo.
# MAGIC - **Spark**: Para el manejo escalable de datos e ingeniería de características.
# MAGIC - **Delta Lake**: Asegura un almacenamiento y recuperación de datos eficiente.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Alcance
# MAGIC
# MAGIC Este notebook abarca la **fase de entrenamiento y evaluación del modelo** dentro del pipeline de detección de fraudes. Se asume que la preprocesamiento de datos y la ingeniería de características se han realizado previamente en un pipeline separado.
# MAGIC

# COMMAND ----------

!pip install mlflow
!pip install pytest

# COMMAND ----------

# MAGIC %md
# MAGIC ## Just for explore the dataset
# MAGIC

# COMMAND ----------

from pyspark.sql import SparkSession

# Crear una sesión de Spark
spark = SparkSession.builder \
    .appName("Read Delta Table") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

# Ruta del archivo Delta
delta_path = "/FileStore/tables/output_delta_table_datapipe_feature_eng_to_train"

# Leer el archivo Delta en un DataFrame
df = spark.read.format("delta").load(delta_path)

# Mostrar las primeras filas del DataFrame
df.show(20, truncate=False)


# COMMAND ----------

df.describe().show()

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
def create_spark_session(app_name="Training Pipeline"):
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
# MAGIC ## Load and Prepare Data
# MAGIC
# MAGIC ### **Propósito:**
# MAGIC Carga datos desde un archivo en formato Delta y prepara las características para el entrenamiento del modelo de detección de fraudes.
# MAGIC
# MAGIC ### **Proceso:**
# MAGIC 1. **Cargar datos:**
# MAGIC    - Lee el archivo Delta ubicado en la ruta especificada (`delta_path`).
# MAGIC    - Verifica que los datos se hayan cargado correctamente.
# MAGIC 2. **Ensambles de características:**
# MAGIC    - Usa `VectorAssembler` para combinar las columnas indicadas en `feature_columns` en una sola columna llamada `features`.
# MAGIC 3. **Conversión de la etiqueta:**
# MAGIC    - Convierte la columna de etiqueta (`label_column`) a tipo entero para que sea compatible con el modelo.
# MAGIC
# MAGIC ### **Output:**
# MAGIC Un DataFrame que incluye:
# MAGIC - **`features`**: Columna con las características combinadas para el modelo.
# MAGIC - **`label_column`**: La columna de etiqueta convertida a tipo entero.
# MAGIC

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col

# Cargar datos desde Delta y ensamblar características
def load_and_prepare_data(spark, delta_path, feature_columns, label_column):
    try:
        # Cargar los datos desde Delta
        df = spark.read.format("delta").load(delta_path)
        logger.info("Data loaded successfully from Delta.")
        
        # Crear columna 'features'
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        df = assembler.transform(df)
        logger.info("Features assembled successfully.")
        
        # Convertir la columna de etiqueta a entero
        df = df.withColumn(label_column, col(label_column).cast("integer"))
        return df
    except Exception as e:
        logger.error("Error loading and preparing data: %s", str(e))
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add Class Weight
# MAGIC
# MAGIC ### **Propósito:**
# MAGIC Aborda el problema de desbalance de clases en el conjunto de datos asignando pesos a cada registro, de manera que las clases minoritarias (fraudes) tengan mayor influencia en el modelo durante el entrenamiento.
# MAGIC
# MAGIC ### **Proceso:**
# MAGIC 1. **Cálculo de totales:**
# MAGIC    - Calcula el número total de registros, transacciones fraudulentas y no fraudulentas en el conjunto de datos.
# MAGIC 2. **Determinación de pesos:**
# MAGIC    - Calcula un peso para cada clase:
# MAGIC      - `fraud_weight`: Peso asignado a las transacciones fraudulentas.
# MAGIC      - `non_fraud_weight`: Peso asignado a las transacciones no fraudulentas.
# MAGIC 3. **Asignación de pesos:**
# MAGIC    - Crea una nueva columna `class_weight` que contiene los pesos correspondientes:
# MAGIC      - `fraud_weight` para registros etiquetados como fraude.
# MAGIC      - `non_fraud_weight` para registros etiquetados como no fraude.
# MAGIC
# MAGIC ### **Output:**
# MAGIC Un DataFrame que incluye:
# MAGIC - **`class_weight`**: Columna con los pesos asignados a cada registro para balancear las clases durante el entrenamiento.
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import when

# Agregar la columna 'class_weight'
def add_class_weight(df, label_column="is_fraud"):
    try:
        total_count = df.count()
        fraud_count = df.filter(col(label_column) == 1).count()
        non_fraud_count = df.filter(col(label_column) == 0).count()

        fraud_weight = total_count / (fraud_count * 2)
        non_fraud_weight = total_count / (non_fraud_count * 2)

        df = df.withColumn(
            "class_weight",
            when(col(label_column) == 1, fraud_weight).otherwise(non_fraud_weight)
        )
        logger.info("Class weights added successfully.")
        return df
    except Exception as e:
        logger.error("Error adding class weights: %s", str(e))
        raise


# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Model with Cross-Validation and MLflow
# MAGIC
# MAGIC ### **Propósito:**
# MAGIC Entrenar un modelo de regresión logística para detectar fraudes en transacciones de tarjetas de crédito, utilizando validación cruzada para optimizar hiperparámetros y rastrear los experimentos en **MLflow**.
# MAGIC
# MAGIC ### **Proceso:**
# MAGIC 1. **Configuración de MLflow:**
# MAGIC    - Establece el experimento de MLflow para rastrear los parámetros, métricas y modelos.
# MAGIC
# MAGIC 2. **Definición del modelo:**
# MAGIC    - Configura un modelo de regresión logística con las siguientes columnas:
# MAGIC      - **`features`**: Características del conjunto de datos.
# MAGIC      - **`is_fraud`**: Etiqueta de clase.
# MAGIC      - **`class_weight`**: Pesos para balancear las clases.
# MAGIC
# MAGIC 3. **Cuadrícula de hiperparámetros:**
# MAGIC    - Define una cuadrícula para el parámetro `regParam` con valores `[0.01, 0.1, 1.0]`.
# MAGIC
# MAGIC 4. **Validación cruzada:**
# MAGIC    - Configura un esquema de validación cruzada con 3 particiones para evaluar las combinaciones de hiperparámetros.
# MAGIC    - Usa el **Área Bajo la Curva (AUC)** como métrica de evaluación.
# MAGIC
# MAGIC 5. **Entrenamiento del modelo:**
# MAGIC    - Ajusta el modelo utilizando validación cruzada y selecciona el mejor modelo según la métrica de AUC.
# MAGIC
# MAGIC 6. **Registro en MLflow:**
# MAGIC    - Registra los parámetros y el modelo entrenado en MLflow.
# MAGIC
# MAGIC ### **Output:**
# MAGIC El modelo de regresión logística entrenado con los hiperparámetros óptimos:
# MAGIC - **`bestModel`**: El mejor modelo seleccionado basado en AUC.
# MAGIC - **Rastreo en MLflow**: Incluye parámetros, métricas y artefactos del modelo.
# MAGIC

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow
import mlflow.spark

# Entrenar el modelo con validación cruzada y MLflow
def train_model_with_cv(df, experiment_name):
    try:
        # Configurar MLflow
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run() as run:
            # Configurar el modelo
            lr = LogisticRegression(featuresCol="features", labelCol="is_fraud", weightCol="class_weight", maxIter=10)

            # Configurar la cuadrícula de hiperparámetros
            paramGrid = ParamGridBuilder() \
                .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
                .build()

            # Configurar la validación cruzada
            evaluator = BinaryClassificationEvaluator(labelCol="is_fraud", metricName="areaUnderROC")
            cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)

            # Entrenar el modelo
            logger.info("Training the model with cross-validation.")
            cvModel = cv.fit(df)

            # Registrar el modelo en MLflow
            mlflow.log_param("RegParam", [0.01, 0.1, 1.0])
            mlflow.spark.log_model(cvModel.bestModel, "fraud_detection_model")
            logger.info("Model logged to MLflow.")
        
        return cvModel.bestModel
    except Exception as e:
        logger.error("Error during model training: %s", str(e))
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluar el Modelo
# MAGIC
# MAGIC ### **Propósito:**
# MAGIC Evaluar el rendimiento del modelo entrenado para detectar fraudes utilizando la métrica **AUC** (Área Bajo la Curva ROC).
# MAGIC
# MAGIC ### **Proceso:**
# MAGIC 1. **Generar Predicciones:**
# MAGIC    - El modelo entrenado se aplica al conjunto de datos de entrada para generar una columna de predicciones.
# MAGIC    
# MAGIC 2. **Configurar el Evaluador:**
# MAGIC    - Se utiliza el evaluador `BinaryClassificationEvaluator` para calcular la métrica de **AUC** (Área Bajo la Curva ROC).
# MAGIC    
# MAGIC 3. **Calcular el AUC:**
# MAGIC    - El evaluador calcula el AUC en función de las predicciones generadas por el modelo.
# MAGIC    
# MAGIC 4. **Registrar el Resultado:**
# MAGIC    - El valor de AUC se registra en los logs para análisis posterior.
# MAGIC
# MAGIC ### **Output:**
# MAGIC - **AUC**: Un valor numérico que representa la calidad del modelo en la detección de fraudes, donde un valor cercano a 1 indica un excelente rendimiento.
# MAGIC

# COMMAND ----------

# Evaluar el modelo
def evaluate_model(model, df):
    try:
        # Generar predicciones
        predictions = model.transform(df)

        # Evaluar el modelo
        evaluator = BinaryClassificationEvaluator(labelCol="is_fraud", metricName="areaUnderROC")
        auc = evaluator.evaluate(predictions)
        logger.info(f"Model evaluation completed. AUC: {auc:.4f}")
        return auc
    except Exception as e:
        logger.error("Error during model evaluation: %s", str(e))
        raise


# COMMAND ----------

# MAGIC %md
# MAGIC # 6. Ejecución Principal 
# MAGIC
# MAGIC - Configura y ejecuta todo el pipeline de detección de fraudes.
# MAGIC - Carga los datos, agrega pesos para balancear clases, entrena el modelo y evalúa su rendimiento.
# MAGIC - Usa MLflow para rastrear los experimentos y resultados.

# COMMAND ----------

if __name__ == "__main__":
    try:
        # Configuración inicial
        logger.info("Starting the fraud detection pipeline.")
        spark = create_spark_session()

        # Rutas y configuración
        delta_path = "/FileStore/tables/output_delta_table_datapipe_feature_eng_to_train"
        model_path = "/dbfs/tmp/fraud_detection_cv_model"
        experiment_name = "/Users/miguel.program.73@gmail.com/fraud_detection_experiment"
        feature_columns = [
            "transaction_velocity",
            "amount_velocity",
            "merchant_category_count",
            "hour_of_day",
            "day_of_week"
        ]
        label_column = "is_fraud"

        # Cargar y preparar los datos
        data = load_and_prepare_data(spark, delta_path, feature_columns, label_column)

        # Agregar la columna 'class_weight'
        data_with_weights = add_class_weight(data, label_column)

        # Entrenar el modelo
        best_model = train_model_with_cv(data_with_weights, experiment_name)

        # Evaluar el modelo
        auc = evaluate_model(best_model, data_with_weights)
        logger.info(f"Pipeline completed successfully. AUC: {auc:.4f}")
        print(f"Model evaluation completed. AUC: {auc:.4f}")

    except Exception as e:
        logger.error("Pipeline execution failed: %s", str(e))
        raise


# COMMAND ----------

# MAGIC %md
# MAGIC ## Descripción de las Pruebas
# MAGIC
# MAGIC ### `test_load_and_prepare_data`
# MAGIC - Verifica que la función cargue los datos desde Delta correctamente.
# MAGIC - Asegura que las columnas `features` y `is_fraud` estén presentes en el DataFrame.
# MAGIC
# MAGIC ### `test_add_class_weight`
# MAGIC - Valida que se genere la columna `class_weight` correctamente.
# MAGIC - Verifica que los pesos sean distintos para las clases de fraude y no fraude.
# MAGIC
# MAGIC ### `test_train_model_with_cv`
# MAGIC - Comprueba que la función entrene un modelo de regresión logística utilizando validación cruzada.
# MAGIC - Valida que el modelo resultante sea del tipo `LogisticRegressionModel`.
# MAGIC
# MAGIC ### `test_evaluate_model`
# MAGIC - Valida que el modelo pueda generar predicciones.
# MAGIC - Verifica que el AUC calculado esté dentro del rango válido de 0 a 1.
# MAGIC
# MAGIC **Nota:** Estas pruebas unitarias, debido a limitaciones de tiempo y a la imposibilidad de ejecutarlas directamente en el mismo notebook, se dejaron de forma hipotética (no las ejecuté como tal, pero aún asi las hice por el requerimiento). Según la metodología de pytest, estas pruebas se ejecutan en archivos `.py`. Por esta razón, decidí priorizar otros pasos del proceso.
# MAGIC

# COMMAND ----------

import pytest

@pytest.fixture(scope="session")
def spark():
    """Crea una sesión de Spark para las pruebas."""
    return SparkSession.builder \
        .appName("TestFraudDetection") \
        .master("local[*]") \
        .getOrCreate()

@pytest.fixture
def sample_data(spark):
    """Crea un DataFrame de prueba."""
    data = [
        ("T1", "C1", 3, 150.0, 5, 12, 3, 0),
        ("T2", "C1", 1, 100.0, 4, 18, 2, 1),
        ("T3", "C2", 2, 200.0, 3, 14, 4, 0)
    ]
    schema = ["transaction_id", "customer_id", "transaction_velocity", "amount_velocity",
              "merchant_category_count", "hour_of_day", "day_of_week", "is_fraud"]
    return spark.createDataFrame(data, schema)

def test_load_and_prepare_data(spark, sample_data):
    """Prueba la función load_and_prepare_data."""
    delta_path = "/tmp/sample_delta"
    sample_data.write.format("delta").mode("overwrite").save(delta_path)

    feature_columns = ["transaction_velocity", "amount_velocity", "merchant_category_count", "hour_of_day", "day_of_week"]
    label_column = "is_fraud"

    df = load_and_prepare_data(spark, delta_path, feature_columns, label_column)
    assert "features" in df.columns
    assert label_column in df.columns

def test_add_class_weight(sample_data):
    """Prueba la función add_class_weight."""
    df = add_class_weight(sample_data, "is_fraud")
    assert "class_weight" in df.columns
    fraud_weights = df.filter(df.is_fraud == 1).select("class_weight").distinct().collect()
    non_fraud_weights = df.filter(df.is_fraud == 0).select("class_weight").distinct().collect()
    assert len(fraud_weights) == 1
    assert len(non_fraud_weights) == 1

def test_train_model_with_cv(sample_data):
    """Prueba la función train_model_with_cv."""
    experiment_name = "test_experiment"
    feature_columns = ["transaction_velocity", "amount_velocity", "merchant_category_count", "hour_of_day", "day_of_week"]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    sample_data = assembler.transform(sample_data)

    df_with_weights = add_class_weight(sample_data, "is_fraud")
    best_model = train_model_with_cv(df_with_weights, experiment_name)
    assert isinstance(best_model, LogisticRegressionModel)

def test_evaluate_model(spark, sample_data):
    """Prueba la función evaluate_model."""
    feature_columns = ["transaction_velocity", "amount_velocity", "merchant_category_count", "hour_of_day", "day_of_week"]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    df = assembler.transform(sample_data)

    lr = LogisticRegression(featuresCol="features", labelCol="is_fraud", weightCol="class_weight", maxIter=10)
    df_with_weights = add_class_weight(df, "is_fraud")
    model = lr.fit(df_with_weights)

    auc = evaluate_model(model, df_with_weights)
    assert isinstance(auc, float)
    assert 0.0 <= auc <= 1.0
