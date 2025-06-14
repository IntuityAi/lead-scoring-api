@echo off
echo Creating deployment package...
tar -czf app.tar.gz app.py requirements.txt Procfile runtime.txt
echo Uploading to Heroku...
heroku slugs:create --app lead-scoreing-api --file app.tar.gz
echo Deployment complete!
pause