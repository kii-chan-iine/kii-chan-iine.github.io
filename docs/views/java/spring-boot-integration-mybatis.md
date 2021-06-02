---
author: 星火燎原@vxhly
title: Spring Boot 整合 Mybatis
categories: [java]
tags: [SpringBoot, Maven, Java, Mybatis, MySQL]
date: 2018-08-22 16:01:15
---

<Boxx changeTime="10000"/>

::: tip 提示
MyBatis 是一款优秀的持久层框架, 它支持定制化 SQL、存储过程以及高级映射。MyBatis 避免了几乎所有的 JDBC 代码和手动设置参数以及获取结果集。这里介绍 Spring Boot 整合 Mybatis 的步骤, 数据库选用 MYSQL。
:::
<!-- more -->

## 修改 pom.xml

``` xml
<dependency>
    <groupId>org.mybatis.spring.boot</groupId>
    <artifactId>mybatis-spring-boot-starter</artifactId>
    <version>1.3.2</version>
</dependency>

<dependency>
    <groupId>mysql</groupId>
    <artifactId>mysql-connector-java</artifactId>
    <scope>runtime</scope>
</dependency>

<dependency>
    <groupId>org.mybatis.generator</groupId>
    <artifactId>mybatis-generator-maven-plugin</artifactId>
    <version>1.3.7</version>
</dependency>
```

## 修改 Spring Boot 配置文件

这里使用 yml 格式的配置文件, 将 application.properties 改名为 application.yml, 下面配置请根据自己的需求而修改

``` 
#
端口
server:
    port: 8080

# 配置数据源
spring:
    datasource:
    url: jdbc: mysql: //127.0.0.1:3306/databaseName?useUnicode=true&characterEncoding=utf8
    username: root
password: root
driver - class - name: com.mysql.jdbc.Driver

# 打印 SQL 语句信息
logging:
    level:
    com.mysql.api: debug

debug: false
```

Spring Boot 会自动加载 application.yml 相关配置, 数据源就会自动注入到s qlSessionFactory 中, sqlSessionFactory 会自动注入到 Mapper 中。

## 使用插件快速生成代码

### 配置 pom.xml

添加 Mybatis-generator 插件

``` xml
<dependency>
    <groupId>org.mybatis.generator</groupId>
    <artifactId>mybatis-generator-maven-plugin</artifactId>
    <version>1.3.7</version>
</dependency>

<build>
    <plugins>
        <plugin>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-maven-plugin</artifactId>
            <configuration>
                <fork>true</fork>
            </configuration>
        </plugin>
        <plugin>
            <groupId>org.mybatis.generator</groupId>
            <artifactId>mybatis-generator-maven-plugin</artifactId>
            <version>1.3.7</version>
            <configuration>
                <configurationFile>src/main/resources/generatorConfig.xml</configurationFile>
                <verbose>true</verbose>
                <overwrite>true</overwrite>
            </configuration>
            <executions>
                <execution>
                    <id>Generate MyBatis Artifacts</id>
                    <goals>
                        <goal>generate</goal>
                    </goals>
                </execution>
            </executions>
            <dependencies>
                <dependency>
                    <groupId>org.mybatis.generator</groupId>
                    <artifactId>mybatis-generator-core</artifactId>
                    <version>1.3.7</version>
                </dependency>
            </dependencies>
        </plugin>
    </plugins>
</build>
```

### 编写配置文件 generatorConfig.xml

在 IntelliJ IDEA 开发环境下, 此文件需要放在 resource 根目录下, mybatis generator 默认加载此目录的配置文件, 以下文件需要根据自己的情况进行配置

``` xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE generatorConfiguration
        PUBLIC "-//mybatis.org//DTD MyBatis Generator Configuration 1.0//EN"
        "http://mybatis.org/dtd/mybatis-generator-config_1_0.dtd">

<generatorConfiguration>
    <!--数据库驱动jar -->
    <classPathEntry
        location="D:\.m2\repository\mysql\mysql-connector-java\5.1.33\mysql-connector-java-5.1.33.jar" />

    <context id="Tables" targetRuntime="MyBatis3">
        <!--去除注释 -->
        <commentGenerator>
            <property name="suppressDate" value="true"/>
            <property name="suppressAllComments" value="true" />
        </commentGenerator>

        <!--数据库连接 -->
        <jdbcConnection driverClass="com.mysql.jdbc.Driver"
            connectionURL="jdbc:mysql://129.1.18.18:3306/ssm_demo" userId="root"
            password="root">
        </jdbcConnection>
        <!--默认false Java type resolver will always use java.math.BigDecimal if 
            the database column is of type DECIMAL or NUMERIC. -->
        <javaTypeResolver>
            <property name="forceBigDecimals" value="false" />
        </javaTypeResolver>

        <!--生成实体类 指定包名 以及生成的地址 （可以自定义地址,但是路径不存在不会自动创建 使用Maven生成在target目录下,会自动创建） -->
        <javaModelGenerator targetPackage="com.myapp.api.entity"
                            targetProject="src/main/java">
            <property name="enableSubPackages" value="true" />
            <property name="trimStrings" value="true" />
        </javaModelGenerator>
        <!--生成SQLMAP文件 -->
        <sqlMapGenerator targetPackage="mapping"
                         targetProject="src/main/resources">
            <property name="enableSubPackages" value="false" />
        </sqlMapGenerator>
        <!-- 生成DAO的包名和位置-->
        <!-- XMLMAPPER生成xml映射文件, ANNOTATEDMAPPER 生成的 dao 采用注解来写sql -->
        <javaClientGenerator type="ANNOTATEDMAPPER"
                             targetPackage="com.myapp.api.mapper"
                             targetProject="src/main/java">
            <property name="enableSubPackages" value="false" />
        </javaClientGenerator>

        <!--对应数据库表 mysql可以加入主键自增 字段命名 忽略某字段等 -->
        <table tableName="" domainObjectName=""
            enableCountByExample="false" enableUpdateByExample="false"
            enableDeleteByExample="false" enableSelectByExample="false"
            selectByExampleQueryId="false" />
    </context>
</generatorConfiguration>
```

### 在 IntelliJ IDEA 配置

配置命令 `mybatis-generator:generate -e`

![Spring Boot](https://vxhly.github.io/assets/spring-boot-11.png)

配置好之后点击旁边的运行按钮

![Spring Boot](https://vxhly.github.io/assets/spring-boot-12.png)

当然也可以在 plugins 中找到 mybatis-generator, 双击运行或右击运行都可

![Spring Boot](https://vxhly.github.io/assets/spring-boot-13.png)
