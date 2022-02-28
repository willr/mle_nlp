import datetime as dt

class SimilarityTest:
    def __init__(self, q1: str=None, q2: str=None, probability: float=-1., rounded: float=-1):
        self.datetime = dt.datetime.now()
        self.formatted_dt = self.datetime.strftime("%b %d %H:%M:%S")
        self.q1 = q1
        self.q2 = q2
        self.probability = probability
        self.rounded = rounded
