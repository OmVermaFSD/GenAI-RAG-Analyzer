package com.documiner;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.openfeign.EnableFeignClients;

@SpringBootApplication
@EnableFeignClients
public class DocManagerApplication {
    public static void main(String[] args) {
        SpringApplication.run(DocManagerApplication.class, args);
    }
}
