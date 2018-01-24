/*
 * Implement all your JavaScript in this file!
 */

$("button").click(function() {

    if (this.value) {

        if ($("#display").data("fromPrevious") == true) { 
            
            resetCalculator($(this).text()); 
                
        } else if (($("#display").data("isPendingFunction") == true) && ($("#display").data("valueOneLocked") == false)) { 
                
            $("#display").data("valueOne", $("#display").val()); 
            $("#display").data("valueOneLocked", true); 
                
            $("#display").val($(this).text()); 
            $("#display").data("valueTwo", $("#display").val()); 
            $("#display").data("valueTwoLocked", true); 
            
              // Clicking a number AGAIN, after first number locked and already value for second number    
        } else if (($("#display").data("isPendingFunction") == true) && ($("#display").data("valueOneLocked") == true)) { 
    
            var curValue = $("#display").val(); 
            var toAdd = $(this).text(); 
    
            var newValue = curValue + toAdd; 
    
            $("#display").val(newValue); 
        
            $("#display").data("valueTwo", $("#display").val()); 
            $("#display").data("valueTwoLocked", true); 
    
        // Clicking on a number fresh    
        } else { 
    
            var curValue = $("#display").val(); 
            if (curValue == "0") { 
                curValue = ""; 
            } 
    
            var toAdd = $(this).text(); 
    
            var newValue = curValue + toAdd; 
    
            $("#display").val(newValue); 
    
        } 

    }

    else if {

        if ($("#display").data("fromPrevious") == true) { 
            resetCalculator($("#display").val()); 
            $("#display").data("valueOneLocked", false); 
            $("#display").data("fromPrevious", false) 
        } 
          
        // Let it be known that a function has been selected 
        var pendingFunction = $(this).text(); 
        $("#display").data("isPendingFunction", true); 
        $("#display").data("thePendingFunction", pendingFunction); 
          
        // Visually represent the current function 
        $(".function-button").removeClass("pendingFunction"); 
        $(this).addClass("pendingFunction"); 
    }

    
    // equation = equation + this.value;
    // $("#display").html(equation);
})


function resetCalculator(curValue) { 
    $("#display").val(curValue); 
    $(".function-button").removeClass("pendingFunction"); 
    $("#display").data("isPendingFunction", false); 
    $("#display").data("thePendingFunction", ""); 
    $("#display").data("valueOneLocked", false); 
    $("#display").data("valueTwoLocked", false); 
    $("#display").data("valueOne", curValue); 
    $("#display").data("valueTwo", 0); 
    $("#display").data("fromPrevious", false); 
}