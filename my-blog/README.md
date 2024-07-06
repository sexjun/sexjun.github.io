```json
{
  "steps": [
    {
      "step": 1,
      "description": "安装 Gatsby CLI",
      "command": "npm install -g gatsby-cli"
    },
    {
      "step": 2,
      "description": "创建一个新的 Gatsby 项目",
      "command": "gatsby new my-blog https://github.com/gatsbyjs/gatsby-starter-blog"
    },
    {
      "step": 3,
      "description": "启动开发服务器",
      "commands": [
        "cd my-blog",
        "gatsby develop"
      ],
      "url": "http://localhost:8000"
    },
    {
      "step": 4,
      "description": "自定义你的博客",
      "notes": "修改 src 文件夹下的内容来定制你的网站。博客文章位于 content/blog 目录下。"
    },
    {
      "step": 5,
      "description": "生成静态文件",
      "command": "gatsby build"
    },
    {
      "step": 6,
      "description": "部署到 GitHub Pages",
      "substeps": [
        {
          "step": 6.1,
          "description": "创建一个新的 GitHub 仓库"
        },
        {
          "step": 6.2,
          "description": "将项目上传到 GitHub",
          "commands": [
            "git init",
            "git add .",
            "git commit -m 'Initial commit'",
            "git branch -M main",
            "git remote add origin https://github.com/yourusername/my-blog.git",
            "git push -u origin main"
          ]
        },
        {
          "step": 6.3,
          "description": "安装 gh-pages 插件",
          "command": "npm install --save-dev gh-pages"
        },
        {
          "step": 6.4,
          "description": "配置 package.json 文件",
          "json": {
            "homepage": "https://yourusername.github.io/my-blog",
            "scripts": {
              "deploy": "gatsby build && gh-pages -d public"
            }
          }
        },
        {
          "step": 6.5,
          "description": "部署到 GitHub Pages",
          "command": "npm run deploy",
          "url": "https://yourusername.github.io/my-blog"
        }
    }
  ]
}

```
