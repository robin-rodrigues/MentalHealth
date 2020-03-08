var express  = require("express");
var router   = express.Router();
var User     = require("../model/user");
var passport = require("passport");
var Diagnostic = require('../model/diagnostic');

router.get("/",function(req,res){
    res.render("index");
});

router.get("/fd",function(req,res){
    res.render("fd");
});

router.get("/guide",function(req,res){
    res.render("guide");
});

//Diagnostic routes
router.get("/diagnostic",function(req,res){
    res.render("diagnostic");
});

router.get("/meditation",function(req,res){
    res.render("meditation");
});

router.post("/diagnostic",function(req,res){
    var newDiagnostic = new Diagnostic({age: req.body.age,
                                        gender: req.body.gender,
                                        family_history: req.body.family_history,
                                        benefits: req.body.benefits,
                                        care_options: req.body.care_options,
                                        anonymity: req.body.anonymity,
                                        leave: req.body.leave,
                                        work_interfere: req.body.work_interfere});
    Diagnostic.create(newDiagnostic, function(err, newlyCreatedDiagnostic){
          if(err){
              console.log(err);
          } else {
              //redirect back to campgrounds page
              console.log(newlyCreatedDiagnostic);
              res.redirect("/name");
          }
      });
});

//======================
//  AUTH ROUTES
//======================
//show register form
router.get("/register",function(req,res){
    res.render('register');
});
//handle signup logic
router.post("/register",function(req,res){
    var newUser = new User({username: req.body.username});
    User.register(newUser, req.body.password, function(err,user){
       if(err){
           req.flash("error", err.message);
           return res.render('register');
       }
       passport.authenticate("local")(req,res,function(){
           req.flash("success","Welcome to CampsiteView " +  user.username);
           res.redirect("/");
       });
    });
});

//show login form
router.get("/login",function(req,res){
    res.render('login');
});
//handling login logic
router.post("/login",passport.authenticate("local",{
    successRedirect:"/",
    failureRedirect:"/login"
 }),function(req,res){
 });

 //logout route
 router.get("/logout",function(req,res){
    req.logout();
    req.flash("success","Logged you out!");
    res.redirect("/");
 });



module.exports = router;