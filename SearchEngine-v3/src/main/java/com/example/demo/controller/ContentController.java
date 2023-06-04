package com.example.demo.controller;

import com.example.demo.service.ContentService;
import de.l3s.boilerpipe.BoilerpipeProcessingException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.xml.sax.SAXException;

import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

@RestController
public class ContentController {

    @Autowired
    private ContentService contentService;

//    @GetMapping("/searchAn/{aid}")
//    public String parsese(Model model, @PathVariable("aid") String aid) throws IOException, IOException {
//
//        System.out.println(aid);
//        List list=contentService.searchAnswer(aid);
//        String str=list.toString();
//        model.addAttribute("hello",list.toString());
//        return "答案： \n"+ str;
//    }

    @GetMapping("/parse/{keyword}")
    // 如何写中文@PathVariable
    public Boolean parse(@PathVariable("keyword") String keyword) throws IOException, IOException, InterruptedException, BoilerpipeProcessingException, SAXException {
//        String convStr="机器学习";
//        String convStr= Arrays.toString(keyword.getBytes(StandardCharsets.UTF_8));
        System.out.println(keyword);
        //return contentService.parseContent(convStr);
        return contentService.parseContent(keyword);
    }


    @GetMapping("/search/{keyword}/{pageNo}/{pageSize}")
    //@CrossOrigin(origin = {"http://127.0.0.1:8080"})//添加来源地址
    public List<Map<String, Object>> search(@PathVariable("keyword") String keyword,
                                            @PathVariable("pageNo") int pageNo,
                                            @PathVariable("pageSize") int pageSize) throws IOException {

        List<Map<String, Object>> list = contentService.searchPage(keyword, pageNo, pageSize);
        return list;
    }

    @GetMapping("/writeQA")
    public Boolean writeQA() throws IOException, IOException {
        return contentService.writeQAContent();
    }

    @PostMapping("/query")
    public List<Map<String, Object>> query(String keyword, int pageNo, int pageSize) throws IOException {

        List<Map<String, Object>> list = contentService.searchPage(keyword, pageNo, pageSize);
        return list;

    }


    @PostMapping("/queryse")
    public List<Map<String, Object>> queryse(String keyword, int pageNo, int pageSize) throws IOException {

        List<Map<String, Object>> list = contentService.searchQA(keyword, pageNo, pageSize);
        return list;

    }
    @PostMapping("/parsese")
    Boolean parsese(String keyword) throws IOException, IOException, InterruptedException, BoilerpipeProcessingException, SAXException {
        Boolean boolvalue = contentService.parseContent(keyword);
        Thread.sleep(1000);
        return boolvalue;

    }
}