### Télécharger le modèle

Le fichier `Model_DVF.pkl` (3,4 Go) est trop volumineux pour GitHub.  
Vous pouvez le télécharger ici :

👉 [Télécharger Model_DVF.pkl sur Google Drive](https://drive.google.com/file/d/1Z79gZJ5R2NzWBHDiZLTxDfOsamm0nkkF/view?usp=drive_link)

### Notebook utilitaire
Un notebook utils.ipynb est inclus dans ce projet pour centraliser les fonctions réutilisables, les scripts d’aide au prétraitement, à l’analyse exploratoire, ou à la visualisation. Ce notebook facilite la maintenance et la modularité du code en regroupant les éléments communs utilisés tout au long du projet.
### Exemple d’utilisation
Dans un autre notebook ou script Python, vous pouvez importer les fonctions du notebook utilitaire comme suit :
### Fonction Convertir_colonnes_booleennes_en_entiers
#### Importation des fonctions définies dans le fichier utils.ipynb
import Utils  
def convert_bool_to_numeric(df):
    for col in df.select_dtypes(include='bool').columns:
        df[col] = df[col].astype(int)
    return df
  ### Application du notebook utils.ipynb  
df_train = Utils.convert_bool_to_numeric(df_train_cleaned)

## Application web interactive avec Streamlit
[![Aperçu du PDF](images/AppDVF.png)](images/AppStreamlit.pdf)
