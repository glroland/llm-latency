import random
import click
from client import execute_inference

MAX_TOKENS = 2000
MAX_WORDS = 2000
MIN_WORDS = 1700

prompts = [
    "Write a story about a dragon who loves to dance.",
    "Describe a futuristic city where robots and humans coexist.",
#    "Imagine a world where everyone has superpowers. What would that look like?",
    "Create a dialogue between a detective and a suspect in a mysterious case.",
    "Write a poem about the changing seasons.",
    "Describe a day in the life of a time traveler.",
    "Imagine a conversation between two aliens visiting Earth for the first time.",
    "Write a letter from a pirate captain to their crew.",
    "Describe a magical forest where the trees can talk.",
    "Create a recipe for a dish that doesn't exist yet.",
    "Write a news article about an unexpected event in a small town.",
    "Imagine a world where animals can speak. What would they say?",
    "Create a character profile for a superhero with an unusual power.",
    "Write a short story set in a post-apocalyptic world.",
    "Describe a dream you had that felt incredibly real.",
 #   "Imagine a world where everyone has to wear a mask. What would that mean for society?",
    "Write a scene from a play set in a coffee shop.",
    "Describe a futuristic sport that combines elements of different games.",
]

def load_file(filename):
    """
    Load a file and return its contents.

    Args:
        filename (str): The name of the file to load.

    Returns:
        str: The contents of the file.
    """
    try:
        with open(filename, 'r') as file:
            file_content = file.read()
            return file_content
    except FileNotFoundError:
        print(f"Error: File not found at path: {filename}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def display_response(response):
    """
    Display the response from the model.

    Args:
        response (dict): The response from the model.
    """
    message = f"{response['test_name']}\t\t{response['ttft']}\t\t{response['duration']}\t\t{len(response['response'])}\t\t{len(response['response'].split())}"
    print(message)

@click.command()
@click.argument('config_filename')
@click.option('--show_responses', is_flag=True, default=False, help='Show responses with results.')
def main(config_filename, show_responses):

    # load configuration
    config_raw = load_file(config_filename)
    if config_raw is None:
        print("Failed to load configuration.")
        return
    config = config_raw.splitlines()

    print("Running tests...")
    print ()

    # process each line of the configuration as a test
    results = []
    for config_entry in config:
        if len(config_entry) > 0:
            # parse config entry
            columns = config_entry.split(",")
            test_name = columns[0].strip()
            url = columns[1].strip()
            token = columns[2].strip()
            model = columns[3].strip()
            
            # build prompt
            prompt = random.choice(prompts)
#            adj_prompt = f"{prompt} Respond with approximately {MAX_WORDS} words.  Never respond with less than {MIN_WORDS}."
            adj_prompt = f"{prompt}  Always respond with between {MIN_WORDS} and {MAX_WORDS} words."

            # execute test with a random prompt
            print ("Running test:", test_name)
            response = execute_inference(test_name, url, token, model, adj_prompt, MAX_TOKENS)
            results.append(response)

    # display results
    print ()
    print("All tests completed.")
    print ()
    print("Test\t\t\tTTFT\t\tDuration\tBytes\t\tWords:")
    print("------------------------------------------------------------------------------")
    for result in results:
        display_response(result)
    print ()
    if show_responses:
        print("LLM responses:")
        print ()
        for result in results:
            print ("Test Name --", result['test_name'])
            print ()
            print ("Prompt --", result['prompt'])
            print ()
            print ("Response --", result['response'])
            print ()

if __name__ == '__main__':
    main()
