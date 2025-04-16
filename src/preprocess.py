import os
import pandas as pad
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import tqdm
import joblib
from collections import OrderedDict


def preprocess_data(data_file, output_dir):
    """
    Exercice : Fonction pour prétraiter les données brutes et les préparer pour l'entraînement de modèles.

    Objectifs :
    1. Charger les données brutes à partir d’un fichier CSV.
    2. Nettoyer les données (par ex. : supprimer les valeurs manquantes).
    3. Encoder les labels catégoriels (colonne `family_accession`) en entiers.
    4. Diviser les données en ensembles d’entraînement, de validation et de test selon une logique définie.
    5. Sauvegarder les ensembles prétraités et des métadonnées utiles.

    Indices :
    - Utilisez `LabelEncoder` pour encoder les catégories.
    - Utilisez `train_test_split` pour diviser les indices des données.
    - Utilisez `to_csv` pour sauvegarder les fichiers prétraités.
    - Calculez les poids de classes en utilisant les comptes des classes.
    """

    # Step 1: Load the data
    print('Loading Data')
    data = pad.read_csv(data_file)

    # Step 2: Handle missing values
    data = data.dropna()

    # Step 3: Encode the 'family_accession' to numeric labels
    label_encoder = LabelEncoder()
    label_encoder.fit(data["family_accession"])
    data["family_accession"] = label_encoder.transform(data["family_accession"])

    # Save the label encoder[]
    joblib.dump(label_encoder,'label_encoder.pkl')
    
    # Save the label mapping to a text file
    with open("label_mapping.txt", "w", encoding="utf-8") as f:
        for i, label in enumerate(label_encoder.classes_):
            f.write(f"{i}: {label}\n")

    
    
    # Step 4: Distribute data pas applicable trop de class différentes
    #For each unique class:
    # - If count == 1: go to test set
    # - If count == 2: 1 to dev, 1 to test
    # - If count == 3: 1 to train, 1 to dev, 1 to test
    # - Else: stratified split (train/dev/test)

    print("Distributing data")
    
    train_indices = []
    dev_indices = []
    test_indices = []
    
    label_to_indices = data.groupby("family_accession").groups
    
    # Logic or assigning indices to train/dev/test
    for index, cls in enumerate(tqdm.tqdm(label_encoder.classes_)): 
        
        indices = list(label_to_indices[index]) 
        
        if len(indices) == 1 : 
            test_indices.append(indices[0])
        
        elif len(indices)==2 :
            test_indices.append(indices[0])
            dev_indices.append(indices[1])
        
        elif len(indices)==3:
            test_indices.append(indices[0])
            dev_indices.append(indices[1])
            test_indices.append(indices[2])
        
        else:
            temp_train, temp_remain = train_test_split(indices, test_size=2/3, random_state=42)
            temp_dev, temp_test = train_test_split(temp_remain, test_size=0.5, random_state=42)
            train_indices.extend(temp_train)
            dev_indices.extend(temp_dev)
            test_indices.extend(temp_test)
      
    # Step 5: Convert index lists to numpy arrays
    train_array = np.array(train_indices)
    dev_array = np.array(dev_indices)
    test_array = np.array(test_indices)
    
    # Step 6: Create DataFrames from the selected indices
    df_train = data.iloc[train_array]
    df_dev = data.iloc[dev_array]
    df_test = data.iloc[test_array]
    
    # Step 7: Drop unused columns: family_id, sequence_name, etc.
    df_train = df_train.drop(['family_id','sequence_name'], axis=1)
    df_dev = df_dev.drop(['family_id','sequence_name'], axis=1)
    df_test = df_test.drop(['family_id','sequence_name'], axis=1)
    
    # Step 8: Save train/dev/test datasets as CSV
    df_train.to_csv(os.path.join(output_dir,"train.csv"), index=False)
    df_dev.to_csv(os.path.join(output_dir,"dev.csv"), index=False)
    df_test.to_csv(os.path.join(output_dir,"test.csv"), index=False)

    # Step 9: Calculate class weights from the training set
    class_counts = df_train['family_accession'].value_counts()
    print(class_counts)
    
    train_labels = df_train["family_accession"].values
    classes = np.unique(train_labels)
    
    class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=train_labels) 
    
    for classe, weight in zip(classes,class_weights):
        print(classe,weight) #on le print 
        
    # Step 10: Normalize weights and scale
    max_weight = max(class_weights)
    normalized_weights = class_weights / max_weight
    

    normalized_weight_dict = {
        int(cls): float(weight)
        for cls, weight in zip(classes, normalized_weights)
    }

    print(normalized_weight_dict)
   
    # Step 11: Save the class weights  
    joblib.dump(normalized_weight_dict, "weight_normalise.txt")

    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess protein data")
    parser.add_argument("--data_file", type=str, required=True, help="Path to train CSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the preprocessed files")
    args = parser.parse_args()

    preprocess_data(args.data_file, args.output_dir)
