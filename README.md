# 番茄病害检测系统 部署说明

本项目基于 Django + PyTorch，支持番茄病害图片智能检测。以下为本地部署步骤。

---

## 1. 环境准备

### 1.1 Python 环境


```bash
python3 -m venv venv
source venv/bin/activate
```

### 1.2 安装依赖

进入项目根目录（`AiCourse`），安装依赖：

>请手动安装主要依赖（pytorch安装合适版本即可）：
```bash
pip install django  torch
```

---

## 2. 数据库迁移

初始化数据库：

```bash
python manage.py makemigrations detectApp
python manage.py migrate
```

---

## 3. 确保使用已经预备的数据库
初始化数据库后，将额外数据库文件 `db.sqlite3` 放在项目根目录下（与 `manage.py` 同级）。


## 4. 模型文件准备

确保模型权重文件已放在：

```
Core/Model/ConvNeXt.pth
```

---

## 5. 启动服务

开发环境下运行：

```bash
python manage.py runserver
```

访问 [http://127.0.0.1:8000/](http://127.0.0.1:8000/) 即可使用。

---

## 7. 目录结构说明

```
AiCourse/
├── manage.py
├── Core/
│   ├── ModelUse.py
│   └── Model/
│       └── ConvNeXt.pth
├── detectApp/
│   ├── models.py
│   ├── views.py
│   ├── templates/
│   │   └── detectApp/
│   │       ├── index.html
│   │       ├── disease_detail.html
│   │       └── base.html
│   └── ...
├── DetectingSystem/
│   ├── settings.py
│   └── urls.py
└── ...
```

---

## 8. 常见问题

- **ModuleNotFoundError: No module named 'Core.ModelUse'**
  - 确保 `Core` 目录下有 `__init__.py` 文件。
  - 在项目根目录（含 manage.py）下运行所有命令。

- **模型加载失败**
  - 检查 `Core/Model/ConvNeXt.pth` 路径和文件是否存在。

