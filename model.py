import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

import numpy as np

def calculate_risk(row, coeffs):
    # Start with the intercept
    score = coeffs['Intercept']
    
    # Add linear terms
    score += coeffs['LANE_WIDTH'] * row['LANE_WIDTH']
    score += coeffs['ADT'] * row['ADT']
    score += coeffs['RAISED_REFUGE'] * row['RAISED_REFUGE']
    score += coeffs['CEI'] * row['CEI']
    
    # Add categorical dummies if used (adjust if your data encodes these)
    if row['ROAD_CLASSIFICATION'] == 'SKELETAL':
        score += coeffs.get('C(ROAD_CLASSIFICATION)[T.SKELETAL]', 0)
    elif row['ROAD_CLASSIFICATION'] == 'ARTERIAL':
        score += coeffs.get('C(ROAD_CLASSIFICATION)[T.ARTERIAL]', 0)
    # ... handle other dummies

    # Convert to predicted probability or expected value
    risk = np.exp(score)  # For NB
    # risk = 1 / (1 + np.exp(-score))  # For logistic regression

    return risk

# Load your dataset
df = pd.read_csv('sample_collision_data.csv')

# Fill missing values (assume 0 where data was blank)
df['DIST_FROM_SENIOR_FACILITY'] = df['DIST_FROM_SENIOR_FACILITY'].fillna(0)

# Treat ROAD_CLASSIFICATION as a categorical variable
df['ROAD_CLASSIFICATION'] = df['ROAD_CLASSIFICATION'].astype('category')

# Fit logistic regression model to predict collision occurrence
model = smf.logit(
    formula='COLLISION_OCCURRED ~ LANE_WIDTH + RAISED_REFUGE + ADT + DIST_FROM_BUS_STOP + PED_COUNT + DIST_FROM_SCHOOL + DIST_FROM_SENIOR_FACILITY + CEI + C(ROAD_CLASSIFICATION)',
    data=df
).fit()

# Show results
print(model.summary())

coeffs = model.params
print(coeffs)

df['RISK_SCORE'] = df.apply(lambda row: calculate_risk(row, coeffs), axis=1)

print(df[['INTERSECTION_ID', 'ROAD_NAME', 'RISK_SCORE']])
