# Generated by Django 3.0.3 on 2020-09-21 17:53

from django.db import migrations, models
import kjn_pricetag.models


class Migration(migrations.Migration):

    dependencies = [
        ('kjn_pricetag', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='moviemodel',
            name='movie',
            field=models.FileField(upload_to=kjn_pricetag.models.return_path),
        ),
    ]
