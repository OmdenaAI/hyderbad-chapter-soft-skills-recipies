import pandas as pd

# import all functions from task_2_flow.py
from task_2_flow import *
 

# Read the skills if we want to run ZSC
skills_path='/workspaces/hyderbad-chapter-soft-skills-recipies/src/data/task-2-inputs/skills_df_v3.csv'
skills = pd.read_csv(skills_path)
candidate_labels = list(skills['Skills category'].unique())

# read data
data_path = '/workspaces/hyderbad-chapter-soft-skills-recipies/src/data/task-2-inputs/wiki_how_20221029_sample_200.csv'
data = pd.read_csv(data_path)


# Run the flow - put ZSC_labels at None if you don't want to run ZSC
disaggregated_data = task_2_pipeline(data, 'wikihow', ZSC_labels=None )

print(disaggregated_data.paragraph)

# Export result
disaggregated_data.to_csv('disaggregated_wiki_sample_data_20221029.csv', index = False)
print('File exported !')