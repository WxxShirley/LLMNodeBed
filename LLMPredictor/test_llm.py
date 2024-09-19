from http import HTTPStatus
from dashscope import Generation



def call_with_messages():
    
    messages=[
            {
                'role': 'user',
                'content': 'hello'
            }]
    gen = Generation()
    response = gen.call(
        'chatglm3-6b',
        messages=messages,
        result_format='message',  # set the result is message format.
    )
    print(response)




if __name__ == '__main__':
    call_with_messages()