
import requests

api_url = 'http://localhost:5000/api/submit'

def build_req(q1: str, q2: str):
    json = {}
    json['q1'] = q1
    json['q2'] = q2
    return json

def build_req1():
    req = []
    q1 = 'What can make Physics easy to learn?'
    q2 = 'How can you make physics easy to learn?'
    req.append(build_req(q1, q2))

    q1 = 'What should I do to be a great geologist?'
    q2 = 'How can I be a good geologist?'
    req.append(build_req(q1, q2))

    q1 = 'Do you believe there is life after death?'
    q2 = 'Is it true that there is life after death?'
    req.append(build_req(q1, q2))

    return req

def build_req2():
    req = []
    q1 = 'How do I read and find my YouTube comments?'
    q2 = 'How can I see all my Youtube comments?'
    req.append(build_req(q1, q2))

    q1 = 'How do I see the color blue?'
    q2 = 'How do I see the color blue?'
    req.append(build_req(q1, q2))

    return req

def build_req3():
    req = []
    q1 = 'What is one coin?'
    q2 = 'What\'s this coin?'
    req.append(build_req(q1, q2))

    return req

def build_req4():
    req = []
    q1 = 'Why do girls want to be friends with the guy they reject?'
    q2 = 'How do guys feel after rejecting a girl?'
    req.append(build_req(q1, q2))

    return req
    
def build_err_bad_keys():
    req = []
    req.append({'toast': 'question1', 'q2': 'what is water?'})
    
    return req

def build_req_no_list():
    req = {'q1': 'what is toast?', 'q2': 'what is water?'}
    return req

if __name__ == "__main__":
    req1 = build_req1()
    req2 = build_req2()
    req3 = build_req3()
    req4 = build_err_bad_keys()
    req5 = build_req4()

    res = requests.post(api_url, json=req1)
    print(res.json())
    res = requests.post(api_url, json=req2)
    print(res.json())
    res = requests.post(api_url, json=req3)
    print(res.json())
    res = requests.post(api_url, json=req4)
    print(res.json())
    res = requests.post(api_url, json=req5)
    print(res.json())