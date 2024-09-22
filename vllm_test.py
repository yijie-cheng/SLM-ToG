import requests
prompt = '''
Please retrieve 3 relations (separated by semicolon) that contribute to the question and rate their contribution on a scale from 0 to 1 (the sum of the scores of 3 relations is 1).
Q: Name the president of the country whose main spoken language was Brahui in 1980?
Topic Entity: Brahui Language
Relations: language.human_language.main_country; language.human_language.language_family; language.human_language.iso_639_3_code; base.rosetta.languoid.parent; language.human_language.writing_system; base.rosetta.languoid.languoid_class; language.human_language.countries_spoken_in; kg.object_profile.prominent_type; base.rosetta.languoid.document; base.ontologies.ontology_instance.equivalent_instances; base.rosetta.languoid.local_name; language.human_language.region
A: 1. {language.human_language.main_country (Score: 0.4))}: This relation is highly relevant as it directly relates to the country whose president is being asked for, and the main country where Brahui language is spoken in 1980.
2. {language.human_language.countries_spoken_in (Score: 0.3)}: This relation is also relevant as it provides information on the countries where Brahui language is spoken, which could help narrow down the search for the president.
3. {base.rosetta.languoid.parent (Score: 0.2)}: This relation is less relevant but still provides some context on the language family to which Brahui belongs, which could be useful in understanding the linguistic and cultural background of the country in question.

Q: Lou Seal is the mascot for the team that last won the World Series when?
Topic Entity: Lou Seal
Relations: dataworld.gardening_hint.last_referenced_by; kg.object_profile.prominent_type; sports.mascot.team; sports.sports_team.team_mascot
A:'''

# prompt = '''Hello.'''
# prompt = "Lou Seal is the mascot for the team that last won the World Series when?"

url = "http://localhost:8000/v1/chat/completions"
headers = {
    "Content-Type": "application/json"
}
data = {
    # "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    # "model": "microsoft/Phi-3-mini-4k-instruct",
    "model": "solidrust/Phi-3-mini-4k-instruct-AWQ",
    "messages": [{"role": "user", "content": prompt}]
}

response = requests.post(url, headers=headers, json=data)
print(response.json())