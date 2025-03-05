# Credit Card Fraud Detection System


## Descripción

Este proyecto implementa un sistema para detectar transacciones fraudulentas con tarjetas de crédito. El sistema está diseñado en Databricks Community Edition, aprovechando sus capacidades principales de procesamiento y aprendizaje automático.

## Ejecución

Sigue el orden secuencial de los notebooks incluidos. Cada notebook contiene las instrucciones de configuración necesarias, junto con la documentación y experimentación correspondiente.

### Orden de ejecución:

1. **D.P. & Feature E. - Doc & Exp - Training**  
   - Procesamiento de datos y creación de características para el entrenamiento del modelo.

2. **1.1. D.P. & Feature E. - Doc & Exp - Inference**  
   - Procesamiento de datos y creación de características para la inferencia.

3. **Model Development - Doc & Exp**  
   - Desarrollo y entrenamiento del modelo de detección de fraudes.

4. **Batch Inference Pipeline - Docs and Exp**  
   - Inferencia por lotes para evaluar nuevas transacciones.

5. **Model Evaluation & Monitoring - Docs and Exp**  
   - Evaluación del modelo y monitoreo de métricas de desempeño.

## Notas

- Cada notebook incluye la documentación detallada sobre su configuración y ejecución.
- Es indispensable seguir el orden indicado para garantizar la correcta ejecución del pipeline.

## Limitaciones

- Este proyecto está limitado a procesamiento por lotes debido a las restricciones de Databricks Community Edition.
- Los modelos se almacenan localmente ya que no está disponible MLflow Model Registry.

---

**Autor:** Miguel Angel Velazquez Romero

**Licencia:** MIT

