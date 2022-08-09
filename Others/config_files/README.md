该文件夹用于存储一些本机电脑与linux服务器里的配置文件，包括settings.json，.bashrc，.vscode等工具，记录软件配置文件、linux command、redis history等，方便新入职时可以快速配置linux服务器和电脑。
统一在实习结束的时候把这些文件做好拷贝。


windows、linux公用配置：
- `settings.json`：从笔记本电脑load进来，打开vscode的settings，调出settings.json复制粘贴，防止配置烦恼，比如latex配置，C++配置，高亮显示，vscode各种颜色条设置等等



windows本地配置：
- `.vscode`：从笔记本C盘把文件夹考进来，在用户里面，然后直接放到windows下面就可以自动把笔记本下过的插件给下好。服务器可能要手动比对着下载。


linux服务器配置：
- `authorized_keys`：放置于home/yby下.ssh文件夹下面，没有自己手动创建文件夹，里面的内容更换成自己本机的私钥，具体生成方法看CSDN收藏文章
- `.gitconfig`：可以通过终端命令设置git的global option生成，应该也可以自己创建文件夹。主要设置user.name和user.email，设置了才能从linux服务器和github同步进行操作
- `.rediscli_history`：在终端使用redis-cli计算的命令
- `.python_history`：在终端使用python的命令
- `.bash_history`：所有的linux终端命令储存，除非同时开了多处的终端没有同时记录。按照里面的命令顺序可以快速把anaconda，cuda，cudnn等一系列东西装好，还有配置虚拟环境，解压缩，安装redis，文件夹拷贝，上传，解压，环境变量设置，还有各种安装包给一次性滚好
- `.bashrc`：linux的环境变量。使用vim修改的环境变量都在这个里面，新的服务器直接配置好环境变量