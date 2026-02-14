"""
Script to generate a synthetic dataset for Cat, Dog, Horse classification
Generates 1200 samples with 15 features
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def create_animal_dataset(n_samples=1200):
    """
    Create a synthetic dataset for animal classification
    
    Features represent characteristics that can distinguish cats, dogs, and horses:
    1. head_size: Average head size (normalized)
    2. body_length: Length of the body
    3. ear_size: Size of ears
    4. ear_shape: Shape characteristic (pointed=cat, floppy=dog, varied=horse)
    5. tail_length: Length of tail
    6. tail_thickness: Thickness of tail
    7. leg_length: Average leg length
    8. leg_count: Number of legs (typically 4)
    9. claw_sharpness: Sharpness of claws
    10. bite_force: Estimated bite force
    11. speed_capability: Running speed capability
    12. teeth_count: Number of teeth
    13. coat_density: Hair/coat density
    14. whisker_presence: Presence of whiskers (cat feature)
    15. hoof_presence: Presence of hooves (horse feature)
    """
    
    # Number of samples per class
    n_per_class = n_samples // 3
    
    # CAT features
    cat_features = {
        'head_size': np.random.normal(7.5, 1.2, n_per_class),
        'body_length': np.random.normal(35, 5, n_per_class),
        'ear_size': np.random.normal(8, 1.5, n_per_class),
        'ear_shape': np.random.normal(8, 1, n_per_class),  # Pointed ears
        'tail_length': np.random.normal(25, 4, n_per_class),
        'tail_thickness': np.random.normal(2, 0.3, n_per_class),
        'leg_length': np.random.normal(15, 2, n_per_class),
        'leg_count': np.full(n_per_class, 4),
        'claw_sharpness': np.random.normal(8.5, 1, n_per_class),  # Sharp claws
        'bite_force': np.random.normal(150, 30, n_per_class),
        'speed_capability': np.random.normal(48, 5, n_per_class),  # km/h
        'teeth_count': np.random.normal(30, 2, n_per_class),
        'coat_density': np.random.normal(6, 1.2, n_per_class),
        'whisker_presence': np.random.normal(9, 1, n_per_class),  # High whisker score
        'hoof_presence': np.random.normal(1, 0.5, n_per_class),  # No hooves
    }
    
    # DOG features
    dog_features = {
        'head_size': np.random.normal(10, 2, n_per_class),
        'body_length': np.random.normal(50, 8, n_per_class),
        'ear_size': np.random.normal(9, 2, n_per_class),
        'ear_shape': np.random.normal(5, 2, n_per_class),  # Varied ear shapes
        'tail_length': np.random.normal(30, 6, n_per_class),
        'tail_thickness': np.random.normal(3, 0.5, n_per_class),
        'leg_length': np.random.normal(20, 3, n_per_class),
        'leg_count': np.full(n_per_class, 4),
        'claw_sharpness': np.random.normal(5, 1.5, n_per_class),  # Less sharp
        'bite_force': np.random.normal(200, 50, n_per_class),
        'speed_capability': np.random.normal(40, 8, n_per_class),
        'teeth_count': np.random.normal(42, 2, n_per_class),
        'coat_density': np.random.normal(7, 1.5, n_per_class),
        'whisker_presence': np.random.normal(3, 1.5, n_per_class),  # Few/no whiskers
        'hoof_presence': np.random.normal(1, 0.5, n_per_class),  # No hooves
    }
    
    # HORSE features
    horse_features = {
        'head_size': np.random.normal(25, 3, n_per_class),
        'body_length': np.random.normal(180, 20, n_per_class),
        'ear_size': np.random.normal(12, 2, n_per_class),
        'ear_shape': np.random.normal(6, 1.5, n_per_class),  # Medium pointed
        'tail_length': np.random.normal(90, 10, n_per_class),
        'tail_thickness': np.random.normal(8, 1, n_per_class),
        'leg_length': np.random.normal(90, 10, n_per_class),
        'leg_count': np.full(n_per_class, 4),
        'claw_sharpness': np.random.normal(3, 1, n_per_class),  # Blunt hooves
        'bite_force': np.random.normal(400, 80, n_per_class),
        'speed_capability': np.random.normal(88, 8, n_per_class),  # Fast runners
        'teeth_count': np.random.normal(44, 2, n_per_class),
        'coat_density': np.random.normal(6.5, 1, n_per_class),
        'whisker_presence': np.random.normal(2, 1, n_per_class),  # Minimal whiskers
        'hoof_presence': np.random.normal(9, 0.5, n_per_class),  # Strong hooves
    }
    
    # Combine all data
    all_data = []
    all_labels = []
    
    # Add cat data
    cat_df = pd.DataFrame(cat_features)
    all_data.append(cat_df)
    all_labels.extend(['Cat'] * n_per_class)
    
    # Add dog data
    dog_df = pd.DataFrame(dog_features)
    all_data.append(dog_df)
    all_labels.extend(['Dog'] * n_per_class)
    
    # Add horse data
    horse_df = pd.DataFrame(horse_features)
    all_data.append(horse_df)
    all_labels.extend(['Horse'] * n_per_class)
    
    # Concatenate all data
    X = pd.concat(all_data, ignore_index=True)
    y = pd.Series(all_labels, name='Animal_Type')
    
    # Combine features and labels
    dataset = pd.concat([X, y], axis=1)
    
    # Shuffle dataset
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Clip values to realistic ranges
    dataset['head_size'] = dataset['head_size'].clip(lower=0)
    dataset['body_length'] = dataset['body_length'].clip(lower=0)
    dataset['ear_size'] = dataset['ear_size'].clip(lower=0)
    dataset['tail_length'] = dataset['tail_length'].clip(lower=0)
    dataset['leg_length'] = dataset['leg_length'].clip(lower=0)
    dataset['claw_sharpness'] = dataset['claw_sharpness'].clip(0, 10)
    dataset['bite_force'] = dataset['bite_force'].clip(lower=0)
    dataset['speed_capability'] = dataset['speed_capability'].clip(lower=0)
    dataset['teeth_count'] = dataset['teeth_count'].clip(lower=0)
    dataset['coat_density'] = dataset['coat_density'].clip(0, 10)
    dataset['whisker_presence'] = dataset['whisker_presence'].clip(0, 10)
    dataset['hoof_presence'] = dataset['hoof_presence'].clip(0, 10)
    
    return dataset

# Generate dataset
print("Generating synthetic dataset for Cat, Dog, Horse classification...")
print("="*70)

dataset = create_animal_dataset(n_samples=1200)

# Display dataset info
print(f"\n✓ Dataset created successfully!")
print(f"Total samples: {len(dataset)}")
print(f"Total features: {len(dataset.columns) - 1}")
print(f"\nClass distribution:")
print(dataset['Animal_Type'].value_counts())

print(f"\nFeature names:")
for i, col in enumerate(dataset.columns[:-1], 1):
    print(f"  {i:2d}. {col}")

print(f"\nDataset Statistics:")
print(dataset.describe().round(2))

print(f"\nMissing values:")
print(dataset.isnull().sum())

# Save dataset
output_path = 'animal_classification_dataset.csv'
dataset.to_csv(output_path, index=False)
print(f"\n✓ Dataset saved to: {output_path}")

# Display first few rows
print(f"\nFirst 5 rows of the dataset:")
print(dataset.head().to_string())

print("\n" + "="*70)
print("Dataset ready for training!")
print("="*70)
