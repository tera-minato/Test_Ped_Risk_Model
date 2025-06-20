import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

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
