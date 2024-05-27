import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

data = pd.DataFrame(data={
    'Weather': ['Sunny', 'Sunny', 'Rainy', 'Rainy', 'Sunny' ],
    'Traffic': ['Heavy', 'Light', 'Heavy', 'Light', 'Heavy'],
    'LateForWork': ['Yes', 'No', 'Yes', 'No', 'Yes']
})

model = BayesianNetwork([('Weather', 'Traffic'), ('Traffic', 'LateForWork')])


model.fit(data, estimator=MaximumLikelihoodEstimator)

for cpd in model.get_cpds():
    print(cpd)

inference = VariableElimination(model)
query_result = inference.query(variables=['LateForWork'], evidence={'Weather': 'Sunny'})
print(query_result)
