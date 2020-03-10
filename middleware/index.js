//all the middleware goes here
// var Campgrounds   = require('../models/campground');
// var Comment       = require('../models/comment');
var middlewareObj = {}

// middlewareObj.checkCampgroundOwnership = function(req,res,next){                //middleware
//         //is user logged in
//     if(req.isAuthenticated()){
//         Campgrounds.findById(req.params.id, function(err,foundCampground){
//             if(err){
//                 req.flash("error","Campgrounds not found");
//                 res.redirect("back");
//             }
//             else{
//                 //does user own the campground
//                 if(foundCampground.author.id.equals(req.user._id)){
//                     next();
//                 }else{
//                     //if not,redirect  
//                     req.flash("error","You do not have permission to do that");
//                     res.redirect("back");
//                 }
//             }
//         });
//     }else{
//         //if not,redirect
//         req.flash("error","You need to be logged in to do that");
//         res.redirect("back");
//     }    
// }
    
// middlewareObj.checkCommentOwnership = function(req,res,next){
//     //is user logged in
//     if(req.isAuthenticated()){
//         Comment.findById(req.params.comment_id, function(err,foundComment){
//             if(err){
//                 res.redirect("back");
//             }
//             else{
//                 //does user own the comment?
//                 if(foundComment.author.id.equals(req.user._id)){
//                    next();
//                 }else{
//                     //if not,redirect  
//                     req.flash("error","You do not have permission to do that");
//                     res.redirect("back");
//                 }
//             }
//         });
//     }else{
//         //if not,redirect
//         req.flash("error","You need to be logged in to do that");
//         res.redirect("back");
//     }    
// }

middlewareObj.isLoggedIn = function(req,res,next){
    if(req.isAuthenticated()){
        return next();
     }
     req.flash("error","You need to be logged in to do that");
     res.redirect("/login");
}    
module.exports = middlewareObj;