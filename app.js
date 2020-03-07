require('dotenv').config();

var express         = require('express'),
    app             = express(),
    bodyParser      = require('body-parser'),
    mongoose        = require('mongoose'),
    flash           = require('connect-flash'),
    passport        = require('passport'),
    LocalStrategy   = require('passport-local'),
    methodOverrride = require('method-override'),
    Campgrounds     = require('./model/campground'),
 // Comment         = require('./models/comment'),
    User            = require('./model/user');
 //   seedDB          = require('./seeds');

var CampgroundsRoutes = require("./routes/campgrounds"),
    // CommentsRoutes    = require("./routes/comments"), 
     IndexRoutes      = require("./routes/index");    

mongoose.connect("mongodb://localhost:27017/yelpcamp",{useNewUrlParser: true});
app.use(bodyParser.urlencoded({extended: true}));
app.set("view engine","ejs");
app.use(express.static(__dirname + "/public"));
app.use(methodOverrride("_method"));
app.use(flash());

//seedDB(); //seed the database

// PASSPORT CONFIG
app.use(require("express-session")({
   secret: "Once again rusty wins!",
   resave:  false,
   saveUninitialzed: false
}));
app.use(passport.initialize());
app.use(passport.session());
passport.use(new LocalStrategy(User.authenticate()));
passport.serializeUser(User.serializeUser());
passport.deserializeUser(User.deserializeUser());

app.use(function(req,res,next){
    res.locals.currentUser = req.user;
    res.locals.error = req.flash("error"); 
    res.locals.success = req.flash("success");
    next();
});

app.use("/", IndexRoutes);
app.use("/campgrounds", CampgroundsRoutes);
// app.use("/campgrounds/:id/comments/", CommentsRoutes);


app.listen(3000,function(){
    console.log(" Server has started");
});
