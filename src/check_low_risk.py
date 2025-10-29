import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('../Data/survey lung cancer.csv')

# Filter LOW_RISK samples
low_risk = df[df['LUNG_CANCER'] == 'NO']
high_risk = df[df['LUNG_CANCER'] == 'YES']

print(f"LOW_RISK Samples: {len(low_risk)} (12.6%)")
print(f"HIGH_RISK Samples: {len(high_risk)} (87.4%)")

print("\n" + "="*70)
print("LOW_RISK PROFILE ANALYSIS")
print("="*70)

print(f"\nAge: Min={low_risk['AGE'].min()}, Max={low_risk['AGE'].max()}, Mean={low_risk['AGE'].mean():.1f}")
print(f"\nGender Distribution:")
print(low_risk['GENDER'].value_counts())

print(f"\n\nSymptom Profile (first 3 LOW_RISK samples):")
print("="*70)

for idx, row in low_risk.head(3).iterrows():
    print(f"\nSample {idx+1}:")
    print(f"  Age: {row['AGE']}, Gender: {row['GENDER']}")
    print(f"  Smoking: {row['SMOKING']}, Yellow Fingers: {row['YELLOW_FINGERS']}")
    print(f"  Anxiety: {row['ANXIETY']}, Chronic Disease: {row['CHRONIC DISEASE']}")
    print(f"  Allergy: {row['ALLERGY ']}, Wheezing: {row['WHEEZING']}")
    print(f"  Alcohol: {row['ALCOHOL CONSUMING']}, Coughing: {row['COUGHING']}")
    print(f"  Shortness of Breath: {row['SHORTNESS OF BREATH']}")
    print(f"  Swallowing Difficulty: {row['SWALLOWING DIFFICULTY']}")
    print(f"  Chest Pain: {row['CHEST PAIN']}")

