import os
import constants as c

def test_save(model: str, tokenizer: str):
    # save the model
    local_model_path = f'./data_ignore/{c.VERSION}'
    cwd = os.getcwd()
    print(f'save model to: {os.path.join(cwd, local_model_path)}')
    with open(local_model_path, 'w') as model_save:
        model_save.write(model)
    print(f'size of tokenizer json: {len(tokenizer)}')
    local_tokenizer_path = f'./data_ignore/tokenizer.{c.VERSION}.json'
    print(f'saving tokenizer to: {os.path.join(cwd, local_tokenizer_path)}')
    with open(local_tokenizer_path, 'w') as token_json:
        token_json.write(tokenizer)

if __name__ == "__main__":
    test_save('model_data:' * 5, 'tokenizer:' * 8)
