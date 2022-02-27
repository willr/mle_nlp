from flask_wtf import FlaskForm

from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Length

class SimilarityForm(FlaskForm):
    q1 = StringField(
        'Text 1',
        [
            DataRequired(),
            Length(max=200, message=('Your message is too long.'))
        ]
    )
    q2 = StringField(
        'Text 2',
        [
            DataRequired(),
            Length(max=200, message=('Your message is too long.'))
        ]
    )
    submit = SubmitField('Submit')