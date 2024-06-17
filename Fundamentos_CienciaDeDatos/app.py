import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings

# Ignorar advertencias
warnings.filterwarnings('ignore')

# 1.2. Recopilación de datos
# Vamos a usar un dataset de ejemplo: Iris dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
data = pd.read_csv(url)

# 1.3. Filtrado de datos
# Supongamos que queremos filtrar solo las especies 'setosa' y 'versicolor'
filtered_data = data[data['species'].isin(['setosa', 'versicolor'])]

# 1.4. Transformación de datos
# Crear una nueva columna que es la suma de 'sepal_length' y 'petal_length'
filtered_data['total_length'] = filtered_data['sepal_length'] + filtered_data['petal_length']

# 1.5. Exploración de datos
print("Primeras filas del dataset filtrado y transformado:")
print(filtered_data.head())
print("\nDescripción estadística:")
print(filtered_data.describe())

# 1.6. Integración de datos
# No tenemos otro dataset para integrar, pero podemos hacer una demostración de agregar un DataFrame vacío
additional_data = pd.DataFrame(columns=['extra_column'])
integrated_data = pd.concat([filtered_data, additional_data], axis=1)

# 1.7. Conceptos de análisis de datos
# Análisis básico de correlación
correlation_matrix = integrated_data.drop(columns=['species']).corr()
print("\nMatriz de correlación:")
print(correlation_matrix)

# 1.8. Clasificación de datos y aprendizaje automático
# Preparamos los datos
X = integrated_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = integrated_data['species'].apply(lambda x: 0 if x == 'setosa' else 1)  # Clasificación binaria

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenamos un clasificador Random Forest
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_scaled, y_train)

y_pred = classifier.predict(X_test_scaled)

# Evaluación del modelo
print("\nInforme de clasificación:")
print(classification_report(y_test, y_pred))
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# 1.9. Comunicación y visualización de datos
# Visualización de la matriz de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación')
plt.show()

# Visualización de la clasificación
plt.figure(figsize=(10, 8))
sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=integrated_data, palette='viridis')
plt.title('Visualización de las especies en el conjunto de datos')
plt.show()
