# Set up repo

1/10/25

1. Make new fork on the [W-doit](https://github.com/W-doit)[talendeur-streamlit](https://github.com/W-doit/talendeur-streamlit) github repo
2. Clone repo
``` bash
cd /Users/maria/Documents/CodeOP/10_Wdoit_Internship

git clone https://github.com/msalvany/talendeur-streamlit.git
```
3. Create a conda environment 
``` bash
cd /Users/maria/Documents/CodeOP/10_Wdoit_Internship/talendeur-streamlit

conda create -n talendeur python=3.10

conda activate talendeur
```
4. Add the original repo as "upstream" and check remotes
```bash
git remote add upstream https://github.com/W-doit/talendeur-streamlit.git

git remote -v

# - origin → your fork
# - upstream → the original W-doit repo
```
5. Install requirements.txt
``` bash
pip install -r requirements.txt
```
6. Try to run the Streamlit app
```bash
streamlit run streamlit_full_version_3.py
```
(to stop the app from working in the terminal: - **CTRL + C**)

7. Open the project in VSC code
``` bash
cd /Users/maria/Documents/CodeOP/10_Wdoit_Internship/talendeur-streamlit

code .
```
8. Activate env
``` bash
conda activate talendeur
```

9. Create .vscode/settings.json so the workspace always uses this env:
``` json
{
  "python.defaultInterpreterPath": "/opt/anaconda3/envs/talendeur/bin/python",
  "terminal.integrated.inheritEnv": true
}
```
10. create a "docs/maria-notes"
 branch to keep my notes and progress
``` bash
# create new branch: docs/maria-notes
git checkout -b docs/maria-notes
# create a docs folder
mkdir docs
# create a new file called setup_notes.md inside the docs/ folder.
touch docs/setup_notes.md
# tell Git to include this file in my next commit
git add docs/setup_notes.md
# record the new file in the branch’s history
git commit -m "Add initial setup notes"
# origin = my fork on GitHub
git push -u origin docs/maria-notes
```
