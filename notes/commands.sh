git config --global user.email "you@example.com"
git config --global user.name "Your Name"

chmod 600 .ssh/id_rsa
git clone git@github.com:Lewington-pitsos/sae.git

python -m venv v
source v/bin/activate
pip install -r requirements.txt

python -m app.cli store sync