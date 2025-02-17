# cnn-breast-cancer-detection

**Universidade de Vigo**
**Pietro Ruzzante**

## Español 
En este trabajo, se ha desarrollado un modelo de Machine Learning para la detección de cáncer de mama a partir de imágenes de mamografías. Se ha utilizado el conjunto de datos de Kaggle RSNA Breast Cancer Detection, que contiene imágenes clasificadas en función de la presencia de tejido canceroso.

Inicialmente, se entrenó un modelo de red neuronal convolucional (CNN) utilizando un conjunto de datos balanceado. Se implementó una arquitectura con filtros pequeños para captar detalles relevantes y se realizaron experimentos eliminando capas convolucionales y densas innecesarias para mejorar el rendimiento. El modelo alcanzó una precisión del 70%, lo que indicaba margen de mejora.

Para optimizar el rendimiento, se aplicó data augmentation, generando nuevas imágenes a partir de pequeñas transformaciones (traslaciones y escalados aleatorios). Luego, se entrenó el modelo con un conjunto de datos desbalanceado, aplicando técnicas para compensar la diferencia en la cantidad de imágenes de cada clase.

En la gráfica de entrenamiento se observa que la precisión aumentó progresivamente, alcanzando valores cercanos al 90%. Además, la función de pérdida disminuyó de manera estable, lo que indica que el modelo logró generalizar bien. Esto demuestra que la data augmentation y el ajuste de la arquitectura fueron claves para mejorar el desempeño del modelo.



## English 
In this project, a Machine Learning model was developed for breast cancer detection using mammography images. The dataset used was Kaggle's RSNA Breast Cancer Detection, which contains images classified based on the presence of cancerous tissue.

Initially, a convolutional neural network (CNN) was trained using a balanced dataset. The model was designed with small filters to capture fine details, and experiments were conducted to optimize performance by removing unnecessary convolutional and dense layers. The model initially achieved 70% accuracy, suggesting room for improvement.

To enhance performance, data augmentation was applied, generating new images with small transformations (random translations and scaling). Then, the model was trained with an imbalanced dataset, applying techniques to compensate for class imbalance.

The training graph shows that accuracy steadily increased, reaching values close to 90%. Additionally, the loss function decreased consistently, indicating that the model successfully generalized. This confirms that data augmentation and architectural optimization were key to improving model performance.
