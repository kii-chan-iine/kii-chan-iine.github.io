---
author: 星火燎原@vxhly
title: SQL 语法基础
categories: [other]
tags: [SQL, Pentest]
date: 2018-01-23 17:04:08
---

<Boxx changeTime="10000"/>

::: tip 前言
整理了下 SQL 语法基础, 适用于 SQL server 和 MySQL, 本博文皆为实例 SQL 语句仅供参考。
:::
<!-- more -->

## 概念

### 基本概念

* 数据库中的行称作 `元组` 、 `记录`
* 数据库中的列称作 `属性` 、 `字段`
* 常见数据库模型 `层次模型`  `网状模型`  `关系模型`  `对象关系模型`  `对象模型`

### 表约束

* 主键约束（Primay Key Coustraint） 唯一性, 非空性
* 唯一约束 （Unique Counstraint）唯一性, 可以空, 但只能有一个
* 检查约束 (Check Counstraint) 对该列数据的范围、格式的限制（如: 年龄、性别等）
* 默认约束 (Default Counstraint) 该数据的默认值
* 外键约束 (Foreign Key Counstraint) 需要建立两表间的关系并引用主表的列

### CURD

* 创建（Create）
* 更新（Update）
* 读取（Retrieve）
* 删除（Delete）

## 数据库

### 创建

``` sql
create database dbxx
on (
  name=dd1,filename="c:\dd\dd1.mdf",
  maxsize=unlimited,filegrowth=10%
),
filegroup xx (
  name=dd2,filename="c:\dd\dd2,mdf",
  maxsize=unlimited
),
(
  name=dd3,filename="d:\dd\dd3.ndf",
  size=50MB,maxsize=unlimited,filegrowth=10MB
)
log on (
  name=LLg,filename="d:\dd\ll.;df",
  size=10MB,maxsize=50MB,filegrowth=20MB
)
```

### 删除

``` sql
use master
drop database dbxx
```

## 创表

``` sql
create table Student
(
  stuNo int,
  stuName varchar(10),
  sex char(2),
  dep varchar(20),
  ID char(18),
  weight decimal(4, 1)
)
```

## 约束

### 主键约束

``` sql
-- 添加约束
alter table Student
add constraint PK_S1 primary key(stuNo)

-- 删除约束
alter table Student
drop constraint PK_S1
```

### 唯一约束

``` sql
-- 添加约束
alter table Student
add constraint UN_S2 union(ID)

-- 删除约束
alter table Student
drop constraint UN_S2
```

### 默认约束

``` sql
-- 添加约束
alter table Student
add constraint DF_S3 default('男') for sex

-- 删除约束
alter table Student
drop constraint DF_S3
```

### 检查约束

``` sql
-- 创建约束
alter table Student
add constraint CK_S4 check(sex='男' or sex='女')

-- 删除约束
alter table Student
drop constraint CK_S4
```

### 外键约束

``` sql
-- 创建约束
alter table Student
add constraint FK_S5 foreign key(stuNo) stuInfo(stuID)

-- 删除约束
alter table Student
drop constraint FK_S5
```

## 删除

``` sql
-- 删表之前一定要先把外键约束先删掉
drop table Student

-- 删除表中的某一列
alter table Student
drop column dep

-- 删除指定数据
delete from Student
where dep="信息系"

-- 逐行删除
delete * from Student

-- 按条件删除
delete * from Student
where sex='男'

-- 快速删除表数据
truncate table Student
```

## 更新

``` sql
-- 新增数据
insert into Student
values (123096832,'张三','男','外语系',350181123401021298)

-- 更新数据
update table Student
set stuName='张三' where stuNo='1013141516'
```

## 查询

``` sql
select * from Student

select all sex from Student
select distinct sex from  Student -- 过滤重复

select top 3 percent * from Student -- 前 3% 条
select top 3 * from Student -- 前 3 条

-- 根据某一列查询
select top 3 with ties weight
from Student
order by weight asc

-- 排序。默认升序
select * from Student order by name
select * from Student order by name desc -- 降序
select * from Student order by name asc -- 升序
```

## 统计

假设 Student 表中共有 20 行记录, 其中有效性别行数为 18

``` sql
select count(*) from Student -- 返回值 20,返回行为 1
select count(sex) from Student -- 返回值 18,不会统计 NULL,返回行为 1
select count(distinct sec) from Student -- 返回值 2,计算不重复的,返回行为 1

select count(weight), weight
from Student group by weight
having weight is not null
```
