# Mr.AshPen
## Mr.ASH Penalized Regression

This a penalized regression formulation of sparse linear regression using the 
adaptive shrinkage (Mr.ASH) prior. 
A VEB formulation of Mr.ASH is available here: [mr.ash.alpha](https://github.com/stephenslab/mr.ash.alpha).
Theory for Mr.AshPen: [Link to Overleaf](https://www.overleaf.com/project/60d0d9301e098e4dbe8e3521) 

### Installation
For development, download this repository and install using `pip`:
```
git clone https://github.com/banskt/mr-ash-pen.git # or use the SSH link
cd mr-ash-pen
pip install -e .
```

### Demonstration
[Link](https://banskt.github.io/iridge-notes/2021/08/24/mrash-penalized-trend-filtering-demo.html) 
to demonstration on simple examples of linear data and trend-filtering data.

### How to use
Functions are not documented yet. Here is only a quick start.
```
from mrashpen.inference.penalized_regression import PenalizedRegression as PLR
plr = PLR(method = 'L-BFGS-B', optimize_w = True, optimize_s = True, is_prior_scaled = True, debug = False)
plr.fit()
```
| Returns | Description |
| --- | --- |
|`plr.coef` | optimized regression coefficients |
|`plr.prior` | optimized Mr.ASH prior mixture coefficients |
|`plr.obj_path` | Value of the objective function for all iterations |
|`plr.theta` | optimized parameter `theta` from the objective function |
|`plr.fitobj` | [OptimizeResult](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult) object from scipy.optimize |
| --- | --- |

### Running tests
Run the unittest from the `/path/to/download/mr-ash-pen` directory.
```
python -m unittest
```
