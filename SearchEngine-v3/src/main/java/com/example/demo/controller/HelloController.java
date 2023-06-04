package com.example.demo.controller;


import com.example.demo.EsDoc;

import com.example.demo.service.ContentService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.ResponseBody;

import java.io.IOException;
import java.util.Collection;
import java.util.List;
import java.util.Map;


//@RestController
@Controller

public class HelloController {

    @Autowired
    private ContentService contentService2;

    @GetMapping({"/","index"})
    public String index(){
        return "index";
    }
    @GetMapping("/jdsearch")
    public String hello2(Model model){
        return "jdsearch";
    }

    @GetMapping("/contentse")
    public String se(Model model){
        return "se-v3";
    }

    @GetMapping("/answer")
    public String answer(Model model){
        return "answer2";
    }

    @GetMapping("/searchAn/{aid}")
    public String parsese(Model model, @PathVariable("aid") String aid) throws IOException, IOException {

        System.out.println(aid);
        List<Map<String, Object>> list=contentService2.searchAnswer(aid);

        model.addAttribute("id",(String)list.get(0).get("id"));
        model.addAttribute("question",(String)list.get(0).get("question"));
//        model.addAttribute("qen",(String)list.get(0).get("qen"));
//        model.addAttribute("qdomain",(String)list.get(0).get("qdomain"));
        model.addAttribute("answer",(String)list.get(0).get("answer"));
        model.addAttribute("all_answers",(Collection<? extends String>) list.get(0).get("all_answers"));
        model.addAttribute("isBing",(Boolean)list.get(0).get("isBing"));
        model.addAttribute("context",(String)list.get(0).get("context"));

        return "answer2";
    }
    @GetMapping("/hello")
    public String hello(Model model){
        model.addAttribute("hello","hello welcome");
        return "test";
    }

    @GetMapping("/hello1")
    @ResponseBody
    public String handle01() throws IOException {
        String str;
        str=EsDoc.searchDoc();
        return str+"\nHello, Spring Boot2!";

    }

    @GetMapping("/getStr")
    @ResponseBody
    public String getStr() throws IOException {
        String str;
        return "\nHello, Spring Boot2!";
    }
}