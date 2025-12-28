#Helper function to create project structure from structure.py
from pathlib import Path
def create_structure(base_path:Path,structure:dict):
    for name,content in structure.items():
        path = Path(base_path,name)
        
        if isinstance(content,dict):
            path.mkdir(parents = True,exist_ok= True) #Level 1 folder
            create_structure(path,content)
        elif isinstance(content,list): #Level 2 Sub folder
            path.mkdir(parents = False,exist_ok= True)
            for file in content: #File
                (path / file).touch(exist_ok=True)
        else:
            path.touch(exist_ok=True)

            
        