# sudo apt-get install git-lfs
# git clone https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx

# eg. prompts = Can you bring up some music for me? I want 3 pop music songs to be played

import onnxruntime_genai as og

model = og.Model('Phi-3-mini-4k-instruct-onnx/cpu_and_mobile/cpu-int4-rtn-block-32')
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()
 
# Set the max length to something sensible by default,
# since otherwise it will be set to the entire context length
search_options = {}
search_options['max_length'] = 2048

chat_template = '<|system|>\n{system_prompt}<|end|>\n<|user|>\n{user_input} <|end|>\n<|assistant|>'

# https://github.com/microsoft/Phi-3CookBook/issues/13#issuecomment-2125309792
system_prompt = '''
You are a virtual assistant capable of handling various tasks using tools designed to interact with different APIs and systems. Based on the user's input, identify and execute the appropriate tool function from the list below. If the user's request cannot be handled by a tool, respond based on your knowledge without generating a tool call. Only return the tool call response if a tool function needs to be invoked.

Available tools:
1. `get_weather_forecast`: Retrieves the weather forecast for a given location. Arguments: city_name.
2. `find_on_spotify`: Searches for tracks, artists, albums, or playlists on Spotify. Arguments: query, type, limit (default limit is 1).

Determine the appropriate tool based on user input and format the tool call as shown in the examples:

Example 1 (Weather Forecast):
- User Input: "What's the weather like in New York City today?"
- Tool Call: {function: {name: get_weather_forecast, arguments: {city_name: "New York City"}}}

Example 2 (Spotify Search):
- User Input: "Find me 2 popular rock tracks."
- Tool Call: {function: {name: find_on_spotify, arguments: {query: "rock", type: "track", limit: 2}}}

Do not include additional information in the tool call response. For all other interactions, maintain a friendly and engaging tone, providing accurate information. Continue to learn from user feedback to improve responses over time.

Example 3 (General Knowledge):
- User Input: "How are GPUs helping AI?"
- Response: "GPUs, or Graphics Processing Units, are particularly well-suited for the parallel processing tasks that are common in AI, especially deep learning. They can process many operations simultaneously, which is beneficial for the matrix and vector calculations that are fundamental to neural network computations. This parallelism allows for faster processing of large datasets and more complex models, which in turn accelerates the training and inference phases of AI applications."
'''
# Integrate the results from tools into your responses as follows:

# 1. If a tool function successfully executes and returns data (e.g., weather forecast), append its result directly in the response message using clear formatting (e.g. "The weather forecast for New York City is ...").
# 2. When invoking tools with parameters, ensure you describe the task context clearly to maintain consistency and understanding. (e.g. "Based on the rock genre on Spotify, the <track> will to be played ...").
# 3. If multiple tool functions could be relevant or no specific tool is identified for a given request, carefully choose one based on user needs, or suggest alternatives if necessary.

# Example function_response: `get_weather_forecast(city="New York City")` -> {temperature: 24, unit: celsius, precipitation:low}
# - Assistant Response Integration: "The weather in New York City has a high of 24°C and low precipitation."
# Example function_response: `find_on_spotify({query: "rock", type: "genre", limit:2})` -> {genre:rock, track_info:[{title: I am so loving..., realease_year: 2023, album:Love is Guilt, artist:Sonu Shin}, {title: Hatred is the only way..., realease_year: 2023, album:Love is Guilt, artist:Sonu Shin}}]
# - Assistant Response Integration: "I found 2 tracks -I am so loving... &  Hatred is the only way... from 'Love is Guilt' Album sung by Sonu Shin on 2023" 

text = input("Input: ")

if not text:
   print("Error, input cannot be empty")
   exit

prompt = f'{chat_template.format(system_prompt=system_prompt, user_input=text)}'

input_tokens = tokenizer.encode(prompt)

params = og.GeneratorParams(model)
params.set_search_options(**search_options)
params.input_ids = input_tokens
generator = og.Generator(model, params)

print("Output: ", end='', flush=True)

try:
   while not generator.is_done():
     generator.compute_logits()
     generator.generate_next_token()

     new_token = generator.get_next_tokens()[0]
     print(tokenizer_stream.decode(new_token), end='', flush=True)
except KeyboardInterrupt:
    print("  --control+c pressed, aborting generation--")

print()
del generator