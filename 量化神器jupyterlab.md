# 量化神器jupyterlab
JupyterLab是一个交互式的开发环境，是jupyter notebook的下一代产品，集成了更多的功能，等其正式版发布，相信那时就是jupyter notebook被取代的时候。  
之前做量化数据的清洗、可视化操作都是使用的Jupyter notebook或者pycharm的数据科学功能，最近接触了一次JupyterLab后，就被它的强大
和便捷所深深的吸引。

## 安装
```
pip install jupyterlab
```
## 插件
安装插件需要基于nodejs环境，所以安装插件前需要先安装nodejs。  
```
conda install -c conda-forge nodejs
```
常用插件安装：
```
# 目录结构显示
jupyter labextension install @jupyterlab/toc

# Voyager 数据优化浏览
jupyter labextension install jupyterlab_voyager

# Drawio 画流程图
jupyter labextension install jupyterlab-drawio

# Lantern数据绘图加强
jupyter labextension install pylantern
jupyter serverextension enable --py lantern
```
然后你就可以开始尽情的享用它的多窗口、可视化、多语言支持、可扩展性  
notebook资源推荐：
首推官方资源列表 [16] ：https://github.com/jupyter/jupyter/wiki/A-gallery-of-interesting-Jupyter-Notebooks  
https://nbviewer.jupyter.org/
