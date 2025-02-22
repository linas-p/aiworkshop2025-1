import ollama

import numpy as np
import pandas as pd


descriptions = np.array([
    "Alice Smith, known for her innovative approach at TechCorp, has been making waves in the engineering field. With a passion for problem solving, she often collaborates with industry leaders. You can contact her at 555-123-4567, and she also mentors aspiring engineers.",
    "Bob Johnson is an accomplished artist whose creative exhibitions in New York have drawn widespread acclaim. He is not only celebrated for his art but also for his contributions to community art programs. For bookings or collaborations, dial 555-234-5678.",
    "Charlie Brown, a famous musician, has a knack for blending classical and modern sounds. His career spans several decades, during which his phone number 555-345-6789 has become a key contact for event organizers. His music continues to inspire many.",
    "In the bustling city of Gotham, Detective Diana Prince stands out for her unmatched investigative skills. Her commitment to justice is evident in every case she handles, and she is available for consultation at 555-456-7890. Her strategic insights have solved numerous high-profile cases.",
    "Agent Ethan Hunt is known for his daring missions and exceptional problem-solving abilities. His covert operations often involve sensitive communications, and his contact number, 555-567-8901, is kept secure among trusted circles. Beyond his secretive work, he is a master strategist.",
    "Fiona Gallagher is an innovative entrepreneur whose startup has revolutionized the tech industry. With a forward-thinking mindset, she continuously drives innovation, and her office can be reached at 555-678-9012 for business inquiries. Her leadership inspires many young professionals.",
    "George Martin, a novelist with a flair for storytelling, has captivated audiences worldwide with his compelling narratives. He occasionally shares insights on literature and culture, and can be contacted at 555-789-0123 for interviews or speaking engagements.",
    "At City High, teacher Hannah Baker has earned a reputation for dedication and excellence in education. Her innovative teaching methods and approachable personality make her a favorite among students. For school-related inquiries, her contact is 555-890-1234.",
    "Mathematician Ian Malcolm, revered for his groundbreaking theories, is a sought-after speaker at academic conferences. His expertise is widely recognized, and for further information or collaborations, reach him at 555-901-2345. His research continues to push boundaries in mathematics.",
    "Julia Roberts, celebrated for her charismatic performances, remains a beloved figure in the entertainment industry. Aside from her acting career, she actively engages with fans and advocates for social causes. You can contact her public relations office at 555-012-3456."
])

prompt = """Instruction:
Analyze the following text and perform the following tasks:

Extract the Person's Name and Surname: Identify and extract any full names (first and last names) present in the text.
Extract the Phone Number: Find any phone number in the format xxx-xxx-xxxx.
Generate a De‑anonymized Text: Create a version of the text where each detected name is replaced with the placeholder <NAME> and each detected phone number is replaced with <PHONE>.
Output Format:
Return a JSON object with the following keys:

"extracted_names": an array of the extracted full names (e.g., ["Alice Smith", "Bob Johnson"]).
"extracted_phones": an array of the extracted phone numbers (e.g., ["555-123-4567"]).
"deanonymised_text": the text with names and phone numbers replaced by <NAME> and <PHONE> respectively.
Example Input Text:
"Alice Smith, known for her work at TechCorp, can be contacted at 555-123-4567. She is a leading engineer."

Example Expected Output:

json
Copy
{
  "extracted_names": ["Alice Smith"],
  "extracted_phones": ["555-123-4567"],
  "deanonymised_text": "<NAME>, known for her work at TechCorp, can be contacted at <PHONE>. She is a leading engineer."
}
Text to Process:

"""

results = []

for description in descriptions:
    response = ollama.chat(model='llama3.2', messages=[
    {
        'role': 'user',
        'content': prompt + description
    },
    ])
    results.append(response['message']['content'])

df = pd.DataFrame(results)
df.to_csv('results.csv')
print(df)