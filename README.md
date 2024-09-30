<br>
<br>
<p align="center">
<img src="logo.png" alt="gamearg logo" style="width:400px;"/>
</p>

# gamearg
A Reasoning Tool for Argumentation Framework, powered by Logic Programming and Game Theory.
If you want to learn more about gamearg from research perspective, check out our papers.
> Xia, Y., Bowers, S., Li, L., & Ludäscher, B. (2024). Reconciling conflicting data curation actions: Transparency through argumentation. International Journal of Digital Curation, 2024.
<br><br>
Ludäscher, B., & Xia, Y. (2023). [Games and Argumentation: Time for a Family Reunion!](https://arxiv.org/pdf/2309.06620.pdf). arXiv preprint arXiv:2309.06620. 
<br><br>
Ludäscher, B., Bowers, S., & Xia, Y. (2023). [Games, Queries, and Argumentation Frameworks: Time for a Family Reunion!](https://ceur-ws.org/Vol-3546/paper06.pdf).
In CEUR Workshop Proceedings (Vol. 3546). CEUR-WS. 




## Usage
### Installing Locally
```
git clone git@github.com:idaks/gamearg.git
cd gamearg
conda env create -f environment.yaml
conda activate gamearg
```

### GitHub Codespace Setup

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/idaks/Games-and-Argumentation)

To set up your environment, simply click on the `Open in GitHub Codespaces` button above. Once the environment setup is complete, you'll be presented with an online version of VScode.

> **Notice:** If you encounter alerts like "Container build failed. Check troubleshooting guide" or "This codespace is currently running in recovery mode due to a configuration error", don't panic. Just press `Ctrl+Shift+P` and type `>Codespaces: Full Rebuild Container` to resolve the issue.

This process will take around 4-8 minutes. **Please do not** press any button until you see something like: `@username  ➜ /workspaces/Games-and-Argumentation (main) $ `

#### For Non-First-Time Usage
You can find the codespace you created at [this link](https://github.com/codespaces).

### Access Notebooks Jupyter

After setting up the local or Codespace environment:

1. Type `jupyter lab` in the Terminal of the VScode online version.
   
   > **Notice:** Sometimes, due to codespace limitations, the terminal may go blank. Simply refreshing your browser should solve the problem.

2. This will lead you to the Jupyter Lab interface, where you can run notebooks

## Developer Installation
The first step is to install the correct version of conda for your operating system
```
conda create --name gamearg python=3.10
conda activate gamearg
git clone the https://github.com/idaks/gamearg
cd gamearg
pip install --editable ".[dev,examples]"
```
Once the package has been installed, we need to install the `pre-commit` to maintain code hygiene. 
```
pre-commit install
```

## License
The software is available under the MIT License.

## Contact
For any queries, please open an issue on GitHub or contact [Yilin Xia](https://yilinxia.com/)

