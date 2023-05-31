PROMPT = (
    'SYSTEM: Your task involves receiving a question from a HUMAN, then extracting the corresponding information from a CSV file in response to this query. '
    'The final result must be structured as a JSON object, containing "name", "property", "value", and "unit" as keys in the dictionary or within a list of dictionaries. '
    'The "unit" is the value inside parentheses in the CSV header, and it should be included without parentheses. '
    'If there is no value for a key, return None. '
    'If a KeyError or IndexError occurs, return {\"name\": null, \"property\": null, \"value\": null, \"unit\": null}. '
    'Path of CSV file is "{{filename}}", Ensure that the file name is correctly referenced in your code. '
    'If the dataframe doesn\'t contain any data for the requested property or requested structure in the Thought step, '
    'You must stop and directly provide the output {\"name\": null, \"property\": null, \"value\": null, \"unit\": null} without any additional steps. '
    'submit the output {\"name\": null, \"property\": null, \"value\": null, \"unit\": null} without further steps.'
    'If it was extracted in JSON format during the observation phase, it will be returned directly without further checking. \n'
    '\n'
    'Example of plan\n'
    'Action: python_repl_ast\n'
    'Action Input:\n'
    "import pandas as pd\n"
    "import json\n"
    "\n"
    'try:\n'
    '    df = pd.read_csv("filename.csv")\n'
    "    row = df.loc[df['name'] == 'ABEXIQ_clean']\n"
    "    value = row['Accessible Surface Area (m^2/g)'].values[0]\n"
    '    result = {\"name\": \"ABEXIQ_clean\", \"property\": \"Accessible Surface Area\", \"value\": value, \"unit\": \"m^2/g\"}\n'
    "except (KeyError, IndexError):\n"
    '    result = {\"name\": None, \"property\": None, \"value\": None, \"unit\": None}\n'
    'json.dumps(result)\n'
    '\n'
    'Result of python_repl_ast:\n'
    '{\"name\": \"ABEXIQ_clean\", \"property\": \"Accessible Surface Area\", \"value\": 534.065, \"unit\": \"m^2/g\"} # if data is existed\n'
    '{\"name\": null, \"property\": null, \"value\": null, \"unit\": null} # if data is not existed\n'
    '\n'
    'HUMAN: {{question}} Answer in JSON format'
)


# PROMPT = (
#     'SYSTEM: Your task involves receiving a question from a HUMAN, then extracting the corresponding information from a CSV file in response to this query. '
#     'The "unit" is the value inside parentheses in the CSV header, and it should be included without parentheses. '
#     'If a KeyError or IndexError occurs, return "There are no data in dataframe". '
#     'Path of CSV file is "{{filename}}", Ensure that the file name is correctly referenced in your code. '
#     'If the dataframe doesn\'t contain any data for the requested property or requested structure in the Thought step, '
#     'You must stop and directly provide the output "There are no data in dataframe" without any additional steps. '
#     '\n'
#     'Example of plan\n'
#     'Action: python_repl_ast\n'
#     'Action Input:\n'
#     "import pandas as pd\n"
#     "import json\n"
#     "\n"
#     'try:\n'
#     '    df = pd.read_csv("filename.csv")\n'
#     "    row = df.loc[df['name'] == 'ABEXIQ_clean']\n"
#     "    value = row['Accessible Surface Area (m^2/g)'].values[0]\n"
#     '    result = {\"name\": \"ABEXIQ_clean\", \"property\": \"Accessible Surface Area\", \"value\": value, \"unit\": \"m^2/g\"}\n'
#     "except (KeyError, IndexError):\n"
#     '    result = {\"name\": None, \"property\": None, \"value\": None, \"unit\": None}\n'
#     'json.dumps(result)\n'
#     '\n'
#     'HUMAN: {{question}}'
# )
