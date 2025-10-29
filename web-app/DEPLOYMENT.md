# Deployment Guide

Complete guide for deploying the Lung Cancer Prediction Web Application to various platforms.

---

## Table of Contents

1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Heroku Deployment](#heroku-deployment)
4. [AWS Deployment](#aws-deployment)
5. [Azure Deployment](#azure-deployment)
6. [Production Considerations](#production-considerations)

---

## Local Development

### Quick Start (Windows)

```cmd
# Option 1: Using start script
start.bat

# Option 2: Manual setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python backend\app.py
```

### Quick Start (Linux/Mac)

```bash
# Option 1: Using start script
chmod +x start.sh
./start.sh

# Option 2: Manual setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python backend/app.py
```

### Using run.py

```bash
# Basic
python run.py

# Custom port
python run.py --port 8000

# Production mode
python run.py --env production --host 0.0.0.0
```

---

## Docker Deployment

### Build and Run with Docker

```bash
# Build image
docker build -t lung-cancer-predictor .

# Run container
docker run -p 5000:5000 \
  -v $(pwd)/../models:/app/models:ro \
  -v $(pwd)/../results:/app/results:ro \
  lung-cancer-predictor
```

### Using Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## Heroku Deployment

### Prerequisites

- Heroku account
- Heroku CLI installed

### Steps

```bash
# 1. Login to Heroku
heroku login

# 2. Create new app
heroku create lung-cancer-predictor-pk

# 3. Add buildpack
heroku buildpacks:add heroku/python

# 4. Create Procfile
echo "web: gunicorn -w 4 -b 0.0.0.0:\$PORT backend.app:create_app()" > Procfile

# 5. Set environment variables
heroku config:set FLASK_ENV=production
heroku config:set SECRET_KEY=$(openssl rand -hex 32)

# 6. Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main

# 7. Open app
heroku open
```

### Post-Deployment

```bash
# View logs
heroku logs --tail

# Scale dynos
heroku ps:scale web=1

# Run migrations (if needed)
heroku run python migrate.py
```

---

## AWS Deployment

### Option 1: AWS Elastic Beanstalk

```bash
# 1. Install EB CLI
pip install awsebcli

# 2. Initialize EB
eb init -p python-3.9 lung-cancer-predictor

# 3. Create environment
eb create lung-cancer-env

# 4. Deploy
eb deploy

# 5. Open app
eb open
```

### Option 2: AWS EC2

```bash
# 1. SSH into EC2 instance
ssh -i your-key.pem ec2-user@your-instance-ip

# 2. Install dependencies
sudo yum update -y
sudo yum install python3 git nginx -y

# 3. Clone repository
git clone https://github.com/your-repo/lung-cancer-predictor.git
cd lung-cancer-predictor/web-app

# 4. Setup virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 5. Setup Gunicorn service
sudo nano /etc/systemd/system/lung-cancer.service
```

**Service File (`/etc/systemd/system/lung-cancer.service`):**

```ini
[Unit]
Description=Lung Cancer Prediction Web Application
After=network.target

[Service]
User=ec2-user
WorkingDirectory=/home/ec2-user/lung-cancer-predictor/web-app
Environment="PATH=/home/ec2-user/lung-cancer-predictor/web-app/venv/bin"
ExecStart=/home/ec2-user/lung-cancer-predictor/web-app/venv/bin/gunicorn -w 4 -b 127.0.0.1:5000 backend.app:create_app()
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# 6. Start service
sudo systemctl start lung-cancer
sudo systemctl enable lung-cancer

# 7. Configure Nginx
sudo nano /etc/nginx/conf.d/lung-cancer.conf
```

**Nginx Config:**

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    location /static {
        alias /home/ec2-user/lung-cancer-predictor/web-app/static;
    }
}
```

```bash
# 8. Restart Nginx
sudo systemctl restart nginx
```

---

## Azure Deployment

### Azure App Service

```bash
# 1. Install Azure CLI
# Follow: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli

# 2. Login
az login

# 3. Create resource group
az group create --name LungCancerRG --location eastus

# 4. Create app service plan
az appservice plan create --name LungCancerPlan \
  --resource-group LungCancerRG \
  --sku B1 --is-linux

# 5. Create web app
az webapp create --resource-group LungCancerRG \
  --plan LungCancerPlan \
  --name lung-cancer-predictor \
  --runtime "PYTHON|3.9"

# 6. Configure startup command
az webapp config set --resource-group LungCancerRG \
  --name lung-cancer-predictor \
  --startup-file "gunicorn -w 4 -b 0.0.0.0:8000 backend.app:create_app()"

# 7. Deploy code
az webapp up --resource-group LungCancerRG \
  --name lung-cancer-predictor
```

---

## Production Considerations

### Security

```python
# config.py - Production settings
class ProductionConfig(Config):
    DEBUG = False
    TESTING = False
    
    # Use strong secret key
    SECRET_KEY = os.environ.get('SECRET_KEY')
    
    # Enable HTTPS only
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Add security headers
    SECURITY_HEADERS = {
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'SAMEORIGIN',
        'X-XSS-Protection': '1; mode=block'
    }
```

### Performance

1. **Use Gunicorn with multiple workers:**
   ```bash
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker backend.app:create_app()
   ```

2. **Enable caching:**
   - Redis for session storage
   - CDN for static files

3. **Database connection pooling** (if using database)

4. **Load balancing** for high traffic

### Monitoring

```python
# Add logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

### Environment Variables

**Production `.env`:**

```bash
FLASK_ENV=production
SECRET_KEY=<strong-random-key>
HOST=0.0.0.0
PORT=5000
LOG_LEVEL=INFO

# Database (if needed)
DATABASE_URL=postgresql://user:pass@host/db

# External services
SENTRY_DSN=<your-sentry-dsn>
```

### SSL/HTTPS

**Using Let's Encrypt with Nginx:**

```bash
# Install Certbot
sudo yum install certbot python3-certbot-nginx -y

# Get certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo certbot renew --dry-run
```

### Backup Strategy

1. **Database backups** (if applicable)
2. **Model versioning**
3. **Configuration backups**
4. **Automated backup scripts**

---

## Health Checks

### Kubernetes Liveness Probe

```yaml
livenessProbe:
  httpGet:
    path: /api/health
    port: 5000
  initialDelaySeconds: 30
  periodSeconds: 10
```

### External Monitoring

- **Uptime Robot:** https://uptimerobot.com
- **Pingdom:** https://www.pingdom.com
- **New Relic:** https://newrelic.com

---

## Scaling

### Horizontal Scaling

```bash
# Heroku
heroku ps:scale web=3

# Kubernetes
kubectl scale deployment lung-cancer --replicas=3
```

### Vertical Scaling

- Increase dyno size (Heroku)
- Change instance type (AWS EC2)
- Upgrade app service plan (Azure)

---

## Troubleshooting

### Common Issues

1. **Port already in use:**
   ```bash
   # Change port
   python run.py --port 8000
   ```

2. **Models not loading:**
   ```bash
   # Check model paths
   ls ../models/
   # Verify config.py paths
   ```

3. **Memory issues:**
   - Reduce number of workers
   - Use model quantization
   - Implement lazy loading

---

## Support

For deployment assistance:
- GitHub Issues: [your-repo/issues]
- Email: support@example.com
- Documentation: [README.md](README.md)

---

**Last Updated:** October 28, 2025

